package org.ea.javallm.layers;

import org.ea.javallm.autograd.GradientChecker;
import org.ea.javallm.autograd.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the Linear layer: output shape correctness for 2D and 3D inputs,
 * parameter accessibility, and gradient correctness via finite-difference checking.
 */
class LinearTest {

    private Random rng;
    private GradientChecker checker;

    @BeforeEach
    void setUp() {
        rng = new Random(42);
        checker = new GradientChecker(1e-5, 1e-4);
    }

    @Test
    void outputShapeCorrectFor2DInput() {
        Linear linear = new Linear(4, 8, rng);
        Tensor input = randomTensor(new int[]{3, 4}, false);

        Tensor output = linear.forward(input);

        assertArrayEquals(new int[]{3, 8}, output.getShape());
    }

    @Test
    void outputShapeCorrectFor3DInput() {
        Linear linear = new Linear(4, 8, rng);
        Tensor input = randomTensor(new int[]{2, 3, 4}, false);

        Tensor output = linear.forward(input);

        assertArrayEquals(new int[]{2, 3, 8}, output.getShape());
    }

    @Test
    void getParametersReturnsWeightAndBias() {
        Linear linear = new Linear(4, 8, rng);
        List<Tensor> params = linear.getParameters();

        assertEquals(2, params.size());
        assertSame(linear.getWeight(), params.get(0));
        assertSame(linear.getBias(), params.get(1));
    }

    @Test
    void weightHasRequiresGrad() {
        Linear linear = new Linear(4, 8, rng);
        assertTrue(linear.getWeight().isRequiresGrad());
        assertTrue(linear.getBias().isRequiresGrad());
    }

    @Test
    void weightShapeIsOutputDimByInputDim() {
        Linear linear = new Linear(4, 8, rng);
        assertArrayEquals(new int[]{8, 4}, linear.getWeight().getShape());
    }

    @Test
    void biasInitializedToZeros() {
        Linear linear = new Linear(4, 8, rng);
        for (double v : linear.getBias().getData()) {
            assertEquals(0.0, v);
        }
    }

    @Test
    void gradientCheckPasses() {
        Linear linear = new Linear(4, 8, new Random(123));
        Tensor input = randomTensor(new int[]{2, 3, 4}, true);
        Tensor weights = randomTensor(new int[]{2, 3, 8}, false);

        List<Tensor> params = linear.getParameters();
        Tensor weight = params.get(0);
        Tensor bias = params.get(1);

        GradientChecker.Result result = checker.check(() -> {
            Tensor output = linear.forward(input);
            Tensor weighted = output.multiply(weights);
            return sumToScalar(weighted);
        }, input, weight, bias);

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
