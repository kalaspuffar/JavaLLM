package org.ea.javallm.model;

import org.ea.javallm.autograd.Tensor;
import org.ea.javallm.layers.Embedding;
import org.ea.javallm.layers.LayerNorm;
import org.ea.javallm.layers.MultiHeadAttention;
import org.ea.javallm.layers.PositionalEncoding;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * GPT-style decoder-only Transformer model.
 *
 * Architecture: Embedding → Positional Encoding → N × TransformerBlock → LayerNorm → Output.
 *
 * The output projection uses weight tying: logits are computed by multiplying the
 * final hidden states by the transposed embedding weight matrix, rather than
 * using a separate Linear layer. This reduces parameters and is standard practice
 * in GPT-2 and later models.
 *
 * Each TransformerBlock uses causal masking to prevent attending to future positions,
 * making this suitable for autoregressive language modeling.
 */
public class DecoderOnlyModel {

    private final Embedding embedding;
    private final PositionalEncoding positionalEncoding;
    private final TransformerBlock[] blocks;
    private final LayerNorm finalNorm;
    private final Tensor causalMask;
    private final int numLayers;

    /**
     * @param vocabSize   number of tokens in the vocabulary
     * @param embedDim    embedding and hidden dimension
     * @param numLayers   number of Transformer blocks
     * @param numHeads    number of attention heads per block
     * @param ffnInnerDim inner dimension of the feed-forward network (typically 4 × embedDim)
     * @param maxSeqLen   maximum sequence length supported
     * @param rng         random number generator for weight initialization
     */
    public DecoderOnlyModel(int vocabSize, int embedDim, int numLayers, int numHeads,
                            int ffnInnerDim, int maxSeqLen, Random rng) {
        this.numLayers = numLayers;
        this.embedding = new Embedding(vocabSize, embedDim, rng);
        this.positionalEncoding = new PositionalEncoding(maxSeqLen, embedDim);

        this.blocks = new TransformerBlock[numLayers];
        for (int i = 0; i < numLayers; i++) {
            blocks[i] = new TransformerBlock(embedDim, numHeads, ffnInnerDim,
                    true, false, rng);
        }

        this.finalNorm = new LayerNorm(embedDim);

        // Precompute the causal mask once for the maximum sequence length
        this.causalMask = MultiHeadAttention.createCausalMask(maxSeqLen);
    }

    /**
     * Forward pass: token IDs → logits.
     *
     * @param tokenIds batch of token ID sequences, shape (batch, seqLen)
     * @return logits tensor of shape (batch, seqLen, vocabSize)
     */
    public Tensor forward(int[][] tokenIds) {
        Tensor indices = tokenIdsToTensor(tokenIds);

        // Embed tokens and add positional encoding
        Tensor x = embedding.forward(indices);
        x = positionalEncoding.forward(x);

        // Pass through Transformer blocks with causal masking
        for (TransformerBlock block : blocks) {
            x = block.forward(x, null, causalMask);
        }

        // Final layer normalization
        x = finalNorm.forward(x);

        // Weight-tied output projection: logits = x @ embedding.weight^T
        Tensor weightTransposed = embedding.weight.transpose(0, 1);

        // Flatten for 2D matmul then restore shape, since matmul requires 2D or 3D
        int batch = tokenIds.length;
        int seqLen = tokenIds[0].length;
        int embedDim = x.size(2);
        int vocabSize = embedding.weight.size(0);

        Tensor xFlat = x.reshape(batch * seqLen, embedDim);
        Tensor logitsFlat = xFlat.matmul(weightTransposed);
        return logitsFlat.reshape(batch, seqLen, vocabSize);
    }

    /**
     * Returns all learnable parameters. The embedding weight appears exactly once
     * (it is reused for the output projection via weight tying, not duplicated).
     */
    public List<Tensor> getParameters() {
        List<Tensor> params = new ArrayList<>();
        params.addAll(embedding.getParameters());
        // positionalEncoding has no learnable parameters
        for (TransformerBlock block : blocks) {
            params.addAll(block.getParameters());
        }
        params.addAll(finalNorm.getParameters());
        return params;
    }

    /**
     * Returns named parameters with hierarchical dot-separated names.
     * Uses insertion-order LinkedHashMap for consistent serialization ordering.
     */
    public Map<String, Tensor> getNamedParameters() {
        Map<String, Tensor> named = new LinkedHashMap<>();
        named.put("embedding.weight", embedding.weight);
        for (int i = 0; i < numLayers; i++) {
            named.putAll(blocks[i].getNamedParameters("block." + i));
        }
        named.put("finalNorm.gamma", finalNorm.getGamma());
        named.put("finalNorm.beta", finalNorm.getBeta());
        return named;
    }

    /**
     * Converts a 2D int array of token IDs to a flat Tensor for embedding lookup.
     */
    private static Tensor tokenIdsToTensor(int[][] tokenIds) {
        int batch = tokenIds.length;
        int seqLen = tokenIds[0].length;
        double[] data = new double[batch * seqLen];
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seqLen; s++) {
                data[b * seqLen + s] = tokenIds[b][s];
            }
        }
        return new Tensor(data, new int[]{batch, seqLen}, false);
    }
}
