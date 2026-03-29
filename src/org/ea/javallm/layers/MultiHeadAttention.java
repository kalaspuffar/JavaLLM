package org.ea.javallm.layers;

import org.ea.javallm.autograd.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Multi-head scaled dot-product attention from "Attention Is All You Need" (Section 3.2).
 *
 * Supports both self-attention (query == keyValue) and cross-attention (different sources).
 * Optional causal masking prevents attending to future positions.
 *
 * Architecture:
 *   1. Project input through W_Q, W_K, W_V linear layers
 *   2. Split into numHeads independent attention heads
 *   3. Compute scaled dot-product attention per head:
 *      Attention(Q, K, V) = softmax(Q @ K^T / sqrt(headDim)) @ V
 *   4. Concatenate heads and project through W_O
 *
 * Since the autograd MatMul only supports up to 3D, we merge the batch and heads
 * dimensions into a single leading dimension for the attention computation.
 */
public class MultiHeadAttention {

    private final Linear queryProjection;
    private final Linear keyProjection;
    private final Linear valueProjection;
    private final Linear outputProjection;
    private final int numHeads;
    private final int headDim;
    private final double scale;

    /**
     * @param embedDim total embedding dimension (must be divisible by numHeads)
     * @param numHeads number of parallel attention heads
     * @param rng      random number generator for weight initialization
     */
    public MultiHeadAttention(int embedDim, int numHeads, Random rng) {
        if (embedDim % numHeads != 0) {
            throw new IllegalArgumentException(
                    "embedDim (" + embedDim + ") must be divisible by numHeads (" + numHeads + ")");
        }
        this.numHeads = numHeads;
        this.headDim = embedDim / numHeads;
        this.scale = 1.0 / Math.sqrt(headDim);

        this.queryProjection = new Linear(embedDim, embedDim, rng);
        this.keyProjection = new Linear(embedDim, embedDim, rng);
        this.valueProjection = new Linear(embedDim, embedDim, rng);
        this.outputProjection = new Linear(embedDim, embedDim, rng);
    }

    /**
     * Self-attention: query and key/value come from the same input.
     *
     * @param input tensor of shape (batch, seqLen, embedDim)
     * @param mask  causal mask tensor, or null for no masking
     * @return tensor of shape (batch, seqLen, embedDim)
     */
    public Tensor forward(Tensor input, Tensor mask) {
        return forward(input, input, mask);
    }

    /**
     * Attention with separate query and key/value sources.
     * For self-attention, pass the same tensor for both.
     * For cross-attention, query comes from the decoder and keyValue from the encoder.
     *
     * @param query    tensor of shape (batch, querySeqLen, embedDim)
     * @param keyValue tensor of shape (batch, kvSeqLen, embedDim)
     * @param mask     attention mask tensor, or null for no masking
     * @return tensor of shape (batch, querySeqLen, embedDim)
     */
    public Tensor forward(Tensor query, Tensor keyValue, Tensor mask) {
        int batch = query.size(0);
        int querySeqLen = query.size(1);
        int kvSeqLen = keyValue.size(1);
        int embedDim = numHeads * headDim;

        // Project to Q, K, V: each (batch, seqLen, embedDim)
        Tensor q = queryProjection.forward(query);
        Tensor k = keyProjection.forward(keyValue);
        Tensor v = valueProjection.forward(keyValue);

        // Split heads: (batch, seqLen, embedDim) → (batch, seqLen, numHeads, headDim)
        //            → transpose to (batch, numHeads, seqLen, headDim)
        //            → merge to (batch*numHeads, seqLen, headDim) for 3D matmul
        q = splitHeads(q, batch, querySeqLen);
        k = splitHeads(k, batch, kvSeqLen);
        v = splitHeads(v, batch, kvSeqLen);


        // Scaled dot-product attention:
        // scores = Q @ K^T / sqrt(headDim): (batch*numHeads, querySeqLen, kvSeqLen)
        Tensor kTransposed = k.transpose(1, 2);
        Tensor scores = q.matmul(kTransposed).scale(scale);

        // Apply mask if provided (e.g., causal mask fills future positions with -1e9).
        // The Mask op requires matching sizes, so we tile a (1, qLen, kvLen) mask
        // across the batch*numHeads dimension if needed.
        if (mask != null) {
            Tensor expandedMask = expandMaskToBatchHeads(mask, batch * numHeads, querySeqLen, kvSeqLen);
            scores = scores.mask(expandedMask, -1e9);
        }

        // Softmax over key dimension
        Tensor weights = scores.softmax(2);

        // Weighted sum of values: (batch*numHeads, querySeqLen, headDim)
        Tensor attended = weights.matmul(v);

        // Merge heads back: (batch*numHeads, querySeqLen, headDim)
        //                  → (batch, numHeads, querySeqLen, headDim)
        //                  → transpose to (batch, querySeqLen, numHeads, headDim)
        //                  → reshape to (batch, querySeqLen, embedDim)
        Tensor merged = mergeHeads(attended, batch, querySeqLen);

        // Final output projection
        return outputProjection.forward(merged);
    }

    /**
     * Splits heads: reshape from (batch, seqLen, embedDim) to (batch*numHeads, seqLen, headDim).
     *
     * Conceptually: reshape → (batch, seqLen, numHeads, headDim)
     *               transpose → (batch, numHeads, seqLen, headDim)
     *               merge batch+heads → (batch*numHeads, seqLen, headDim)
     *
     * Since our Transpose op handles physical data copies, we can chain
     * reshape → transpose → reshape through the autograd graph.
     */
    private Tensor splitHeads(Tensor input, int batch, int seqLen) {
        // (batch, seqLen, numHeads * headDim) → (batch, seqLen, numHeads, headDim)
        Tensor reshaped = input.reshape(batch, seqLen, numHeads, headDim);
        // → (batch, numHeads, seqLen, headDim)
        Tensor transposed = reshaped.transpose(1, 2);
        // → (batch * numHeads, seqLen, headDim) for 3D matmul compatibility
        return transposed.reshape(batch * numHeads, seqLen, headDim);
    }

    /**
     * Merges heads: reshape from (batch*numHeads, seqLen, headDim) to (batch, seqLen, embedDim).
     *
     * Reverses the splitHeads operation.
     */
    private Tensor mergeHeads(Tensor input, int batch, int seqLen) {
        // (batch * numHeads, seqLen, headDim) → (batch, numHeads, seqLen, headDim)
        Tensor reshaped = input.reshape(batch, numHeads, seqLen, headDim);
        // → (batch, seqLen, numHeads, headDim)
        Tensor transposed = reshaped.transpose(1, 2);
        // → (batch, seqLen, numHeads * headDim)
        return transposed.reshape(batch, seqLen, numHeads * headDim);
    }

    /**
     * Expands a mask to match the (batchHeads, querySeqLen, kvSeqLen) attention score shape.
     *
     * The stored mask may be larger than the actual sequence lengths (e.g., a causal
     * mask created for maxSeqLen when the actual sequence is shorter). In that case,
     * the top-left (querySeqLen, kvSeqLen) submatrix is extracted row by row before
     * tiling across the batch*heads dimension.
     */
    private static Tensor expandMaskToBatchHeads(Tensor mask, int batchHeads, int querySeqLen, int kvSeqLen) {
        int requiredSize = batchHeads * querySeqLen * kvSeqLen;
        int[] maskShape = mask.getShape();
        // Must check shape, not just total size: a (1, 16, 16) mask has the same
        // element count as (16, 4, 4) but a completely different data layout.
        if (maskShape.length == 3
                && maskShape[0] == batchHeads
                && maskShape[1] == querySeqLen
                && maskShape[2] == kvSeqLen) {
            return mask;
        }

        double[] maskData = mask.getData();
        int maskKvDim = mask.getShape()[2];
        int sliceSize = querySeqLen * kvSeqLen;

        // Extract the top-left (querySeqLen, kvSeqLen) submatrix from the mask,
        // which may be stored as (1, maskSeqLen, maskKvDim) where maskSeqLen >= querySeqLen.
        double[] slice = new double[sliceSize];
        for (int row = 0; row < querySeqLen; row++) {
            System.arraycopy(maskData, row * maskKvDim, slice, row * kvSeqLen, kvSeqLen);
        }

        // Tile the extracted slice across all batch*heads
        double[] tiled = new double[requiredSize];
        for (int b = 0; b < batchHeads; b++) {
            System.arraycopy(slice, 0, tiled, b * sliceSize, sliceSize);
        }
        return new Tensor(tiled, new int[]{batchHeads, querySeqLen, kvSeqLen}, false);
    }

    /**
     * Creates a causal (lower-triangular) mask for autoregressive attention.
     *
     * Returns a tensor of shape (1, seqLen, seqLen) where mask[0][i][j] = 1.0 if j <= i,
     * 0.0 otherwise. Pass this to forward() — it will be tiled across batch*numHeads internally.
     *
     * @param seqLen sequence length
     * @return causal mask tensor
     */
    public static Tensor createCausalMask(int seqLen) {
        double[] maskData = new double[seqLen * seqLen];
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                maskData[i * seqLen + j] = (j <= i) ? 1.0 : 0.0;
            }
        }
        return new Tensor(maskData, new int[]{1, seqLen, seqLen}, false);
    }

    /**
     * Returns learnable parameters from all four projection layers.
     */
    public List<Tensor> getParameters() {
        List<Tensor> params = new ArrayList<>();
        params.addAll(queryProjection.getParameters());
        params.addAll(keyProjection.getParameters());
        params.addAll(valueProjection.getParameters());
        params.addAll(outputProjection.getParameters());
        return params;
    }
}
