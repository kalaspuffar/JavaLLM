package org.ea.javallm.model;

import org.ea.javallm.autograd.Tensor;
import org.ea.javallm.layers.Embedding;
import org.ea.javallm.layers.LayerNorm;
import org.ea.javallm.layers.Linear;
import org.ea.javallm.layers.MultiHeadAttention;
import org.ea.javallm.layers.PositionalEncoding;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Full encoder-decoder Transformer model.
 *
 * Architecture:
 * <ul>
 *   <li><b>Encoder:</b> Source Embedding → Positional Encoding → N × EncoderBlock → encoder output</li>
 *   <li><b>Decoder:</b> Target Embedding → Positional Encoding → N × DecoderBlock (with cross-attention
 *       to encoder output) → LayerNorm → Linear output projection</li>
 * </ul>
 *
 * Uses separate source and target embeddings to support different vocabulary sizes.
 * The output projection is a separate Linear layer (no weight tying), which allows
 * srcVocabSize ≠ tgtVocabSize.
 *
 * Positional encoding is shared between encoder and decoder since it depends only
 * on position and embedding dimension, not on the vocabulary.
 */
public class EncoderDecoderModel {

    private final Embedding srcEmbedding;
    private final Embedding tgtEmbedding;
    private final PositionalEncoding positionalEncoding;
    private final TransformerBlock[] encoderBlocks;
    private final TransformerBlock[] decoderBlocks;
    private final LayerNorm finalNorm;
    private final Linear outputProjection;
    private final Tensor causalMask;
    private final int numLayers;

    /**
     * @param srcVocabSize source vocabulary size
     * @param tgtVocabSize target vocabulary size
     * @param embedDim     embedding and hidden dimension
     * @param numLayers    number of encoder and decoder blocks (each)
     * @param numHeads     number of attention heads per block
     * @param ffnInnerDim  inner dimension of the feed-forward network
     * @param maxSeqLen    maximum sequence length supported
     * @param rng          random number generator for weight initialization
     */
    public EncoderDecoderModel(int srcVocabSize, int tgtVocabSize, int embedDim,
                               int numLayers, int numHeads, int ffnInnerDim,
                               int maxSeqLen, Random rng) {
        this.numLayers = numLayers;
        this.srcEmbedding = new Embedding(srcVocabSize, embedDim, rng);
        this.tgtEmbedding = new Embedding(tgtVocabSize, embedDim, rng);
        this.positionalEncoding = new PositionalEncoding(maxSeqLen, embedDim);

        // Encoder blocks: no causal mask, no cross-attention
        this.encoderBlocks = new TransformerBlock[numLayers];
        for (int i = 0; i < numLayers; i++) {
            encoderBlocks[i] = new TransformerBlock(embedDim, numHeads, ffnInnerDim,
                    false, false, rng);
        }

        // Decoder blocks: causal mask + cross-attention to encoder output
        this.decoderBlocks = new TransformerBlock[numLayers];
        for (int i = 0; i < numLayers; i++) {
            decoderBlocks[i] = new TransformerBlock(embedDim, numHeads, ffnInnerDim,
                    true, true, rng);
        }

        this.finalNorm = new LayerNorm(embedDim);
        this.outputProjection = new Linear(embedDim, tgtVocabSize, rng);

        // Precompute causal mask for decoder self-attention
        this.causalMask = MultiHeadAttention.createCausalMask(maxSeqLen);
    }

    /**
     * Forward pass: source token IDs + target token IDs → logits.
     *
     * @param srcTokenIds source token ID sequences, shape (batch, srcSeqLen)
     * @param tgtTokenIds target token ID sequences, shape (batch, tgtSeqLen)
     * @return logits tensor of shape (batch, tgtSeqLen, tgtVocabSize)
     */
    public Tensor forward(int[][] srcTokenIds, int[][] tgtTokenIds) {
        // Encode source
        Tensor srcIndices = tokenIdsToTensor(srcTokenIds);
        Tensor encoderOutput = srcEmbedding.forward(srcIndices);
        encoderOutput = positionalEncoding.forward(encoderOutput);

        for (TransformerBlock block : encoderBlocks) {
            encoderOutput = block.forward(encoderOutput, null, null);
        }

        // Decode target with cross-attention to encoder output
        Tensor tgtIndices = tokenIdsToTensor(tgtTokenIds);
        Tensor x = tgtEmbedding.forward(tgtIndices);
        x = positionalEncoding.forward(x);

        for (TransformerBlock block : decoderBlocks) {
            x = block.forward(x, encoderOutput, causalMask);
        }

        // Final norm and output projection
        x = finalNorm.forward(x);
        return outputProjection.forward(x);
    }

    /**
     * Returns all learnable parameters from all components.
     */
    public List<Tensor> getParameters() {
        List<Tensor> params = new ArrayList<>();
        params.addAll(srcEmbedding.getParameters());
        params.addAll(tgtEmbedding.getParameters());
        for (TransformerBlock block : encoderBlocks) {
            params.addAll(block.getParameters());
        }
        for (TransformerBlock block : decoderBlocks) {
            params.addAll(block.getParameters());
        }
        params.addAll(finalNorm.getParameters());
        params.addAll(outputProjection.getParameters());
        return params;
    }

    /**
     * Returns named parameters with hierarchical dot-separated names.
     * Encoder and decoder block parameters have distinct prefixes.
     */
    public Map<String, Tensor> getNamedParameters() {
        Map<String, Tensor> named = new LinkedHashMap<>();
        named.put("srcEmbedding.weight", srcEmbedding.weight);
        named.put("tgtEmbedding.weight", tgtEmbedding.weight);
        for (int i = 0; i < numLayers; i++) {
            named.putAll(encoderBlocks[i].getNamedParameters("encoder.block." + i));
        }
        for (int i = 0; i < numLayers; i++) {
            named.putAll(decoderBlocks[i].getNamedParameters("decoder.block." + i));
        }
        named.put("finalNorm.gamma", finalNorm.getGamma());
        named.put("finalNorm.beta", finalNorm.getBeta());
        named.put("outputProjection.weight", outputProjection.getWeight());
        named.put("outputProjection.bias", outputProjection.getBias());
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
