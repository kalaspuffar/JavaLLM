package org.ea.javallm.model;

import org.ea.javallm.autograd.Tensor;
import org.ea.javallm.layers.FeedForward;
import org.ea.javallm.layers.LayerNorm;
import org.ea.javallm.layers.MultiHeadAttention;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * A single Transformer block with pre-norm residual connections.
 *
 * Configurable for three operating modes via constructor booleans:
 * <ul>
 *   <li><b>Encoder mode</b> (hasCausalMask=false, hasCrossAttention=false):
 *       unmasked self-attention → feed-forward</li>
 *   <li><b>Decoder mode</b> (hasCausalMask=true, hasCrossAttention=false):
 *       causal self-attention → feed-forward</li>
 *   <li><b>Decoder with cross-attention</b> (hasCausalMask=true, hasCrossAttention=true):
 *       causal self-attention → cross-attention to encoder output → feed-forward</li>
 * </ul>
 *
 * Each sublayer follows the pre-norm pattern: output = x + sublayer(LayerNorm(x)).
 * Pre-norm is more training-stable than post-norm and is standard in GPT-2 and later.
 */
public class TransformerBlock {

    private final LayerNorm selfAttnNorm;
    private final MultiHeadAttention selfAttn;
    private final LayerNorm crossAttnNorm;
    private final MultiHeadAttention crossAttn;
    private final LayerNorm ffnNorm;
    private final FeedForward ffn;
    private final boolean hasCausalMask;
    private final boolean hasCrossAttention;

    /**
     * @param embedDim          embedding dimension
     * @param numHeads          number of attention heads
     * @param ffnInnerDim       inner dimension of the feed-forward network
     * @param hasCausalMask     if true, self-attention uses a causal (lower-triangular) mask
     * @param hasCrossAttention if true, includes a cross-attention sublayer
     * @param rng               random number generator for weight initialization
     */
    public TransformerBlock(int embedDim, int numHeads, int ffnInnerDim,
                            boolean hasCausalMask, boolean hasCrossAttention, Random rng) {
        this.hasCausalMask = hasCausalMask;
        this.hasCrossAttention = hasCrossAttention;

        this.selfAttnNorm = new LayerNorm(embedDim);
        this.selfAttn = new MultiHeadAttention(embedDim, numHeads, rng);

        if (hasCrossAttention) {
            this.crossAttnNorm = new LayerNorm(embedDim);
            this.crossAttn = new MultiHeadAttention(embedDim, numHeads, rng);
        } else {
            this.crossAttnNorm = null;
            this.crossAttn = null;
        }

        this.ffnNorm = new LayerNorm(embedDim);
        this.ffn = new FeedForward(embedDim, ffnInnerDim, rng);
    }

    /**
     * Forward pass through the Transformer block.
     *
     * @param x              input tensor of shape (batch, seqLen, embedDim)
     * @param encoderOutput  encoder output for cross-attention, or null if not used
     * @param causalMask     precomputed causal mask, or null for unmasked attention
     * @return output tensor of shape (batch, seqLen, embedDim)
     */
    public Tensor forward(Tensor x, Tensor encoderOutput, Tensor causalMask) {
        // Pre-norm self-attention with residual: x = x + selfAttn(norm(x))
        Tensor selfAttnMask = hasCausalMask ? causalMask : null;
        Tensor selfAttnOut = selfAttn.forward(selfAttnNorm.forward(x), selfAttnMask);
        x = x.add(selfAttnOut);

        // Optional cross-attention with residual: x = x + crossAttn(norm(x), encoderOutput)
        if (hasCrossAttention && crossAttn != null && encoderOutput != null) {
            Tensor crossAttnOut = crossAttn.forward(
                    crossAttnNorm.forward(x), encoderOutput, null);
            x = x.add(crossAttnOut);
        }

        // Pre-norm feed-forward with residual: x = x + ffn(norm(x))
        Tensor ffnOut = ffn.forward(ffnNorm.forward(x));
        x = x.add(ffnOut);

        return x;
    }

    /**
     * Returns all learnable parameters from all sublayers.
     */
    public List<Tensor> getParameters() {
        List<Tensor> params = new ArrayList<>();
        params.addAll(selfAttnNorm.getParameters());
        params.addAll(selfAttn.getParameters());
        if (hasCrossAttention && crossAttnNorm != null) {
            params.addAll(crossAttnNorm.getParameters());
            params.addAll(crossAttn.getParameters());
        }
        params.addAll(ffnNorm.getParameters());
        params.addAll(ffn.getParameters());
        return params;
    }

    /**
     * Returns named parameters with dot-separated hierarchical names.
     *
     * @param prefix the prefix to prepend to parameter names (e.g., "block.0")
     * @return ordered map of name → tensor pairs
     */
    public Map<String, Tensor> getNamedParameters(String prefix) {
        Map<String, Tensor> named = new LinkedHashMap<>();
        addNamed(named, prefix + ".selfAttnNorm", selfAttnNorm);
        addNamedAttn(named, prefix + ".selfAttn", selfAttn);
        if (hasCrossAttention && crossAttnNorm != null) {
            addNamed(named, prefix + ".crossAttnNorm", crossAttnNorm);
            addNamedAttn(named, prefix + ".crossAttn", crossAttn);
        }
        addNamed(named, prefix + ".ffnNorm", ffnNorm);
        addNamedFfn(named, prefix + ".ffn", ffn);
        return named;
    }

    private static void addNamed(Map<String, Tensor> map, String prefix, LayerNorm norm) {
        map.put(prefix + ".gamma", norm.getGamma());
        map.put(prefix + ".beta", norm.getBeta());
    }

    private static void addNamedAttn(Map<String, Tensor> map, String prefix,
                                     MultiHeadAttention attn) {
        List<Tensor> params = attn.getParameters();
        // MHA returns: W_Q weight, W_Q bias, W_K weight, W_K bias,
        //              W_V weight, W_V bias, W_O weight, W_O bias
        String[] names = {
                ".W_Q.weight", ".W_Q.bias", ".W_K.weight", ".W_K.bias",
                ".W_V.weight", ".W_V.bias", ".W_O.weight", ".W_O.bias"
        };
        for (int i = 0; i < params.size(); i++) {
            map.put(prefix + names[i], params.get(i));
        }
    }

    private static void addNamedFfn(Map<String, Tensor> map, String prefix, FeedForward ffn) {
        List<Tensor> params = ffn.getParameters();
        // FeedForward returns: linear1 weight, linear1 bias, linear2 weight, linear2 bias
        String[] names = {
                ".linear1.weight", ".linear1.bias", ".linear2.weight", ".linear2.bias"
        };
        for (int i = 0; i < params.size(); i++) {
            map.put(prefix + names[i], params.get(i));
        }
    }
}
