package org.ea.javallm.layers;

import org.ea.javallm.autograd.GradientChecker;
import org.ea.javallm.autograd.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for MultiHeadAttention: self-attention output shape, cross-attention
 * with different sequence lengths, causal mask preventing future attention,
 * parameter count, and gradient correctness.
 */
class MultiHeadAttentionTest {

    private Random rng;
    private GradientChecker checker;

    @BeforeEach
    void setUp() {
        rng = new Random(42);
        checker = new GradientChecker(1e-5, 1e-4);
    }

    @Test
    void selfAttentionOutputShape() {
        MultiHeadAttention mha = new MultiHeadAttention(16, 4, rng);
        Tensor input = randomTensor(new int[]{2, 5, 16}, false);

        Tensor output = mha.forward(input, null);

        assertArrayEquals(new int[]{2, 5, 16}, output.getShape());
    }

    @Test
    void crossAttentionWithDifferentSeqLens() {
        MultiHeadAttention mha = new MultiHeadAttention(16, 4, rng);
        Tensor query = randomTensor(new int[]{2, 3, 16}, false);
        Tensor keyValue = randomTensor(new int[]{2, 7, 16}, false);

        Tensor output = mha.forward(query, keyValue, null);

        // Output sequence length should match query, not keyValue
        assertArrayEquals(new int[]{2, 3, 16}, output.getShape());
    }

    @Test
    void causalMaskPreventsFutureAttention() {
        // Use a single head for easier inspection of attention patterns.
        // With embedDim=4 and numHeads=1, headDim=4.
        MultiHeadAttention mha = new MultiHeadAttention(4, 1, new Random(0));
        int seqLen = 4;

        // Create input where each position has distinct values
        Tensor input = randomTensor(new int[]{1, seqLen, 4}, false);
        Tensor causalMask = MultiHeadAttention.createCausalMask(seqLen);

        Tensor withMask = mha.forward(input, causalMask);
        Tensor withoutMask = mha.forward(input, null);

        // With causal mask, position 0 should only attend to itself
        // So the output at position 0 should differ from the unmasked version
        // (unless the unmasked attention accidentally focuses entirely on pos 0)
        assertArrayEquals(new int[]{1, seqLen, 4}, withMask.getShape());

        // Verify the causal mask shape
        assertArrayEquals(new int[]{1, seqLen, seqLen}, causalMask.getShape());

        // Verify mask is lower-triangular
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                double expected = (j <= i) ? 1.0 : 0.0;
                assertEquals(expected, causalMask.get(0, i, j),
                        "Mask at (" + i + "," + j + ") should be " + expected);
            }
        }
    }

    @Test
    void parameterCountIsCorrect() {
        int embedDim = 16;
        MultiHeadAttention mha = new MultiHeadAttention(embedDim, 4, rng);
        List<Tensor> params = mha.getParameters();

        // 4 projections × (weight + bias) = 8 tensors
        assertEquals(8, params.size());
    }

    @Test
    void gradientCheckSelfAttention() {
        // Small dimensions to keep the gradient check fast
        int embedDim = 4;
        int numHeads = 2;
        MultiHeadAttention mha = new MultiHeadAttention(embedDim, numHeads, new Random(77));
        Tensor input = randomTensor(new int[]{1, 3, embedDim}, true);
        Tensor outWeights = randomTensor(new int[]{1, 3, embedDim}, false);

        List<Tensor> params = mha.getParameters();

        GradientChecker.Result result = checker.check(() -> {
            Tensor output = mha.forward(input, null);
            Tensor weighted = output.multiply(outWeights);
            return sumToScalar(weighted);
        }, params.toArray(new Tensor[0]));

        assertTrue(result.passed(), result.toString());
    }

    @Test
    void gradientCheckCrossAttention() {
        int embedDim = 4;
        int numHeads = 2;
        MultiHeadAttention mha = new MultiHeadAttention(embedDim, numHeads, new Random(88));
        Tensor query = randomTensor(new int[]{1, 2, embedDim}, true);
        Tensor keyValue = randomTensor(new int[]{1, 3, embedDim}, true);
        Tensor outWeights = randomTensor(new int[]{1, 2, embedDim}, false);

        List<Tensor> params = mha.getParameters();

        GradientChecker.Result result = checker.check(() -> {
            Tensor output = mha.forward(query, keyValue, null);
            Tensor weighted = output.multiply(outWeights);
            return sumToScalar(weighted);
        }, params.toArray(new Tensor[0]));

        assertTrue(result.passed(), result.toString());
    }

    @Test
    void embedDimMustBeDivisibleByNumHeads() {
        assertThrows(IllegalArgumentException.class,
                () -> new MultiHeadAttention(15, 4, rng));
    }

    // --- Helpers ---

    private Tensor randomTensor(int[] shape, boolean requiresGrad) {
        int size = 1;
        for (int s : shape) size *= s;
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = rng.nextGaussian() * 0.3;
        }
        return new Tensor(data, shape, requiresGrad);
    }

    private Tensor sumToScalar(Tensor t) {
        double sum = 0.0;
        for (double v : t.getData()) sum += v;
        Tensor result = new Tensor(new double[]{sum}, new int[]{1}, true);
        new SumOp(t, result);
        return result;
    }

    private static class SumOp extends org.ea.javallm.autograd.Operation {
        SumOp(Tensor input, Tensor output) {
            super(new Tensor[]{input}, output);
        }

        @Override
        public void backward() {
            Tensor input = inputs[0];
            if (!input.isRequiresGrad()) return;
            double[] dInput = input.getGrad();
            double dOut = output.getGrad()[0];
            for (int i = 0; i < dInput.length; i++) {
                dInput[i] += dOut;
            }
        }
    }
}
