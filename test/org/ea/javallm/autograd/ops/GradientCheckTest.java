package org.ea.javallm.autograd.ops;

import org.ea.javallm.autograd.GradientChecker;
import org.ea.javallm.autograd.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Gradient check tests for all 12 operations using finite-difference verification.
 */
class GradientCheckTest {

    private GradientChecker checker;
    private Random rng;

    @BeforeEach
    void setUp() {
        checker = new GradientChecker(1e-5, 1e-4);
        rng = new Random(42);
    }

    private Tensor randomTensor(int[] shape, boolean requiresGrad) {
        int size = 1;
        for (int s : shape) size *= s;
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = rng.nextGaussian() * 0.5;
        }
        return new Tensor(data, shape, requiresGrad);
    }

    /** Helper: reduce a tensor to a scalar by summing all elements. */
    private Tensor sumToScalar(Tensor t) {
        double sum = 0.0;
        for (double v : t.getData()) sum += v;
        Tensor ones = new Tensor(onesArray(t.size()), t.getShape(), false);
        // Use Multiply + manual sum via a "dot product" approach
        // Actually, let's just reshape and matmul with ones vector, but simpler:
        // Use element-wise multiply with ones then reduce
        // Simplest: create a custom reduction
        Tensor result = new Tensor(new double[]{sum}, new int[]{1}, t.isRequiresGrad());
        new SumOp(t, result);
        return result;
    }

    private static double[] onesArray(int size) {
        double[] arr = new double[size];
        java.util.Arrays.fill(arr, 1.0);
        return arr;
    }

    /**
     * Simple sum operation for reducing tensors to scalar in tests.
     * dInput = dOutput (broadcast to all elements).
     */
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

    // --- MatMul ---

    @Test
    void gradCheckMatMul2D() {
        Tensor a = randomTensor(new int[]{3, 4}, true);
        Tensor b = randomTensor(new int[]{4, 5}, true);

        GradientChecker.Result result = checker.check(() -> {
            Tensor c = a.matmul(b);
            return sumToScalar(c);
        }, a, b);

        assertTrue(result.passed(), result.toString());
    }

    @Test
    void gradCheckMatMul3D() {
        Tensor a = randomTensor(new int[]{2, 3, 4}, true);
        Tensor b = randomTensor(new int[]{2, 4, 5}, true);

        GradientChecker.Result result = checker.check(() -> {
            Tensor c = a.matmul(b);
            return sumToScalar(c);
        }, a, b);

        assertTrue(result.passed(), result.toString());
    }

    // --- Add ---

    @Test
    void gradCheckAddSameShape() {
        Tensor a = randomTensor(new int[]{3, 4}, true);
        Tensor b = randomTensor(new int[]{3, 4}, true);

        GradientChecker.Result result = checker.check(() -> {
            Tensor c = a.add(b);
            return sumToScalar(c);
        }, a, b);

        assertTrue(result.passed(), result.toString());
    }

    @Test
    void gradCheckAddBroadcast() {
        Tensor a = randomTensor(new int[]{2, 3, 4}, true);
        Tensor bias = randomTensor(new int[]{4}, true);

        GradientChecker.Result result = checker.check(() -> {
            Tensor c = a.add(bias);
            return sumToScalar(c);
        }, a, bias);

        assertTrue(result.passed(), result.toString());
    }

    // --- Multiply ---

    @Test
    void gradCheckMultiply() {
        Tensor a = randomTensor(new int[]{3, 4}, true);
        Tensor b = randomTensor(new int[]{3, 4}, true);

        GradientChecker.Result result = checker.check(() -> {
            Tensor c = a.multiply(b);
            return sumToScalar(c);
        }, a, b);

        assertTrue(result.passed(), result.toString());
    }

    // --- Scale ---

    @Test
    void gradCheckScale() {
        Tensor a = randomTensor(new int[]{3, 4}, true);

        GradientChecker.Result result = checker.check(() -> {
            Tensor c = a.scale(0.5);
            return sumToScalar(c);
        }, a);

        assertTrue(result.passed(), result.toString());
    }

    // --- Softmax ---

    @Test
    void gradCheckSoftmax() {
        Tensor a = randomTensor(new int[]{3, 5}, true);

        GradientChecker.Result result = checker.check(() -> {
            Tensor s = a.softmax(1);
            return sumToScalar(s);
        }, a);

        // Softmax rows sum to 1, so sum of all elements = numRows.
        // The gradient of sum(softmax(x)) w.r.t. x is zero because
        // each row sums to 1 regardless of input. Use a weighted sum instead.
        Tensor weights = randomTensor(new int[]{3, 5}, false);

        GradientChecker.Result result2 = checker.check(() -> {
            Tensor s = a.softmax(1);
            Tensor weighted = Multiply.forward(s, weights);
            return sumToScalar(weighted);
        }, a);

        assertTrue(result2.passed(), result2.toString());
    }

    // --- ReLU ---

    @Test
    void gradCheckReLU() {
        // Use values away from zero to avoid discontinuity at zero
        Tensor a = randomTensor(new int[]{3, 4}, true);
        double[] data = a.getData();
        for (int i = 0; i < data.length; i++) {
            if (Math.abs(data[i]) < 0.1) data[i] = 0.5;
        }

        GradientChecker.Result result = checker.check(() -> {
            Tensor c = a.relu();
            return sumToScalar(c);
        }, a);

        assertTrue(result.passed(), result.toString());
    }

    // --- Transpose ---

    @Test
    void gradCheckTranspose() {
        Tensor a = randomTensor(new int[]{3, 4}, true);

        GradientChecker.Result result = checker.check(() -> {
            Tensor t = a.transpose(0, 1);
            return sumToScalar(t);
        }, a);

        assertTrue(result.passed(), result.toString());
    }

    // --- Reshape ---

    @Test
    void gradCheckReshape() {
        Tensor a = randomTensor(new int[]{2, 6}, true);

        GradientChecker.Result result = checker.check(() -> {
            Tensor r = a.reshape(3, 4);
            return sumToScalar(r);
        }, a);

        assertTrue(result.passed(), result.toString());
    }

    // --- LayerNormOp ---

    @Test
    void gradCheckLayerNorm() {
        Tensor a = randomTensor(new int[]{2, 3, 4}, true);
        Tensor gamma = new Tensor(onesArray(4), new int[]{4}, true);
        Tensor beta = new Tensor(new double[4], new int[]{4}, true);

        // Use a weighted sum to make it non-trivial
        Tensor weights = randomTensor(new int[]{2, 3, 4}, false);

        GradientChecker.Result result = checker.check(() -> {
            Tensor normalized = a.layerNorm(gamma, beta, 1e-5);
            Tensor weighted = Multiply.forward(normalized, weights);
            return sumToScalar(weighted);
        }, a, gamma, beta);

        assertTrue(result.passed(), result.toString());
    }

    // --- CrossEntropy ---

    @Test
    void gradCheckCrossEntropy() {
        Tensor logits = randomTensor(new int[]{3, 5}, true);
        Tensor targets = new Tensor(new double[]{1, 3, 0}, new int[]{3}, false);

        GradientChecker.Result result = checker.check(() -> {
            return logits.crossEntropy(targets);
        }, logits);

        assertTrue(result.passed(), result.toString());
    }

    // --- EmbeddingLookup ---

    @Test
    void gradCheckEmbeddingLookup() {
        Tensor weights = randomTensor(new int[]{10, 4}, true);
        Tensor indices = new Tensor(new double[]{2, 5, 2}, new int[]{3}, false);

        // Use a weighted sum to get a non-trivial scalar
        Tensor outWeights = randomTensor(new int[]{3, 4}, false);

        GradientChecker.Result result = checker.check(() -> {
            Tensor embedded = weights.embeddingLookup(indices);
            Tensor weighted = Multiply.forward(embedded, outWeights);
            return sumToScalar(weighted);
        }, weights);

        assertTrue(result.passed(), result.toString());
    }

    // --- Mask ---

    @Test
    void gradCheckMask() {
        Tensor a = randomTensor(new int[]{3, 4}, true);
        // Mask: keep all except column 3
        double[] maskData = new double[]{
                1, 1, 1, 0,
                1, 1, 1, 0,
                1, 1, 1, 0
        };
        Tensor maskTensor = new Tensor(maskData, new int[]{3, 4}, false);

        // Use a small fill value to avoid floating-point precision issues in
        // finite-difference computation — large fill values like -1e9 cause
        // catastrophic cancellation when computing (loss+ - loss-) / 2eps
        GradientChecker.Result result = checker.check(() -> {
            Tensor masked = a.mask(maskTensor, -100.0);
            return sumToScalar(masked);
        }, a);

        assertTrue(result.passed(), result.toString());
    }
}
