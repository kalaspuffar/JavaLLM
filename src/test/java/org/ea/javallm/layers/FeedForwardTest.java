package org.ea.javallm.layers;

import org.ea.javallm.autograd.GradientChecker;
import org.ea.javallm.autograd.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for FeedForward: output shape correctness and gradient flow
 * through the Linear-ReLU-Linear composition.
 */
class FeedForwardTest {

    private Random rng;
    private GradientChecker checker;

    @BeforeEach
    void setUp() {
        rng = new Random(42);
        checker = new GradientChecker(1e-5, 1e-4);
    }

    @Test
    void outputShapePreservesInputShape() {
        FeedForward ff = new FeedForward(8, 32, rng);
        Tensor input = randomTensor(new int[]{2, 3, 8}, false);

        Tensor output = ff.forward(input);

        assertArrayEquals(new int[]{2, 3, 8}, output.getShape());
    }

    @Test
    void getParametersContainsBothLinearLayers() {
        FeedForward ff = new FeedForward(8, 32, rng);
        List<Tensor> params = ff.getParameters();

        // linear1: weight + bias, linear2: weight + bias = 4 tensors
        assertEquals(4, params.size());
    }

    @Test
    void gradientCheckPasses() {
        FeedForward ff = new FeedForward(4, 16, new Random(99));
        Tensor input = randomTensor(new int[]{2, 3, 4}, true);

        // Use ReLU-safe values to avoid zero-gradient discontinuity
        double[] data = input.getData();
        for (int i = 0; i < data.length; i++) {
            if (Math.abs(data[i]) < 0.1) data[i] = 0.5;
        }

        Tensor outWeights = randomTensor(new int[]{2, 3, 4}, false);
        List<Tensor> params = ff.getParameters();

        GradientChecker.Result result = checker.check(() -> {
            Tensor output = ff.forward(input);
            Tensor weighted = output.multiply(outWeights);
            return sumToScalar(weighted);
        }, params.toArray(new Tensor[0]));

        assertTrue(result.passed(), result.toString());
    }

    // --- Helpers ---

    private Tensor randomTensor(int[] shape, boolean requiresGrad) {
        int size = 1;
        for (int s : shape) size *= s;
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = rng.nextGaussian() * 0.5;
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
