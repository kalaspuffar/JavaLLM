package org.ea.javallm.autograd;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Tensor core functionality: storage, indexing, shape queries,
 * factory methods, gradient tracking, and backward pass.
 */
class TensorTest {

    // --- 6.1: Stride computation and index() ---

    @Test
    void strideComputation1D() {
        Tensor t = new Tensor(new double[]{1, 2, 3, 4, 5}, new int[]{5}, false);
        assertArrayEquals(new int[]{1}, t.getStrides());
        assertEquals(3, t.index(3));
    }

    @Test
    void strideComputationAndIndex2D() {
        Tensor t = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, false);
        assertArrayEquals(new int[]{3, 1}, t.getStrides());
        // index(0,1) = 0*3 + 1*1 = 1
        assertEquals(1, t.index(0, 1));
        // index(1,2) = 1*3 + 2*1 = 5
        assertEquals(5, t.index(1, 2));
    }

    @Test
    void strideComputationAndIndex3D() {
        Tensor t = Tensor.zeros(2, 3, 4);
        assertArrayEquals(new int[]{12, 4, 1}, t.getStrides());
        // index(1,2,3) = 1*12 + 2*4 + 3*1 = 23
        assertEquals(23, t.index(1, 2, 3));
    }

    @Test
    void strideComputationAndIndex4D() {
        Tensor t = Tensor.zeros(2, 3, 4, 5);
        assertArrayEquals(new int[]{60, 20, 5, 1}, t.getStrides());
        // index(1,2,3,4) = 1*60 + 2*20 + 3*5 + 4*1 = 60+40+15+4 = 119
        assertEquals(119, t.index(1, 2, 3, 4));
    }

    @Test
    void indexWrongDimensionCountThrows() {
        Tensor t = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, false);
        assertThrows(IllegalArgumentException.class, () -> t.index(0));
        assertThrows(IllegalArgumentException.class, () -> t.index(0, 1, 2));
    }

    @Test
    void constructorRejectsDataShapeMismatch() {
        assertThrows(IllegalArgumentException.class,
                () -> new Tensor(new double[]{1, 2, 3}, new int[]{2, 3}, false));
    }

    // --- 6.2: get()/set() round-trips and factories ---

    @Test
    void getReturnsCorrectElements() {
        Tensor t = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, false);
        assertEquals(2.0, t.get(0, 1));
        assertEquals(6.0, t.get(1, 2));
    }

    @Test
    void setAndGetRoundTrip() {
        Tensor t = Tensor.zeros(3, 4);
        t.set(42.0, 1, 2);
        assertEquals(42.0, t.get(1, 2));
        // Verify other elements remain zero
        assertEquals(0.0, t.get(0, 0));
        assertEquals(0.0, t.get(2, 3));
    }

    @Test
    void zerosFactory() {
        Tensor t = Tensor.zeros(2, 3);
        assertArrayEquals(new int[]{2, 3}, t.getShape());
        assertEquals(6, t.size());
        assertFalse(t.isRequiresGrad());
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(0.0, t.get(i, j));
            }
        }
    }

    @Test
    void randnFactory() {
        Random rng = new Random(42);
        double scale = 0.1;
        Tensor t = Tensor.randn(new int[]{3, 4}, rng, scale);
        assertArrayEquals(new int[]{3, 4}, t.getShape());
        assertEquals(12, t.size());
        assertFalse(t.isRequiresGrad());

        // Verify values are within reasonable range for N(0, 0.01) —
        // with scale=0.1 values should almost always be in [-0.5, 0.5]
        double[] data = t.getData();
        for (double v : data) {
            assertTrue(Math.abs(v) < 1.0,
                    "Value " + v + " seems too large for scale=" + scale);
        }
    }

    @Test
    void shapeQueries() {
        Tensor t = Tensor.zeros(2, 3, 4);
        assertEquals(24, t.size());
        assertEquals(2, t.size(0));
        assertEquals(3, t.size(1));
        assertEquals(4, t.size(2));
        assertEquals(3, t.dims());
    }

    @Test
    void gradientAllocatedWhenRequired() {
        Tensor t = new Tensor(new double[12], new int[]{3, 4}, true);
        assertNotNull(t.getGrad());
        assertEquals(12, t.getGrad().length);
        for (double g : t.getGrad()) {
            assertEquals(0.0, g);
        }
    }

    @Test
    void noGradientWhenNotRequired() {
        Tensor t = new Tensor(new double[12], new int[]{3, 4}, false);
        assertNull(t.getGrad());
    }

    @Test
    void nameSetterAndGetter() {
        Tensor t = Tensor.zeros(2);
        assertNull(t.getName());
        t.setName("W_Q");
        assertEquals("W_Q", t.getName());
    }

    @Test
    void leafTensorHasNullCreator() {
        Tensor t = new Tensor(new double[]{1, 2, 3}, new int[]{3}, true);
        assertNull(t.getCreator());
    }

    @Test
    void zeroGradResetsGradient() {
        Tensor t = new Tensor(new double[]{1, 2, 3}, new int[]{3}, true);
        t.getGrad()[0] = 5.0;
        t.getGrad()[1] = 3.0;
        t.zeroGrad();
        for (double g : t.getGrad()) {
            assertEquals(0.0, g);
        }
    }

    // --- 6.3: backward() on a hand-built graph ---

    /**
     * Builds: a(scalar) and b(scalar) → MultiplyOp → output(scalar)
     * output = a * b
     * d(output)/da = b, d(output)/db = a
     */
    @Test
    void backwardPropagatesGradientsThroughSimpleGraph() {
        Tensor a = new Tensor(new double[]{3.0}, new int[]{1}, true);
        Tensor b = new Tensor(new double[]{4.0}, new int[]{1}, true);

        // Forward: output = a * b
        double[] outputData = new double[]{a.getData()[0] * b.getData()[0]};
        Tensor output = new Tensor(outputData, new int[]{1}, true);

        // Wire the graph via a concrete test Operation
        new ScalarMultiplyOp(a, b, output);

        output.backward();

        // d(a*b)/da = b = 4.0
        assertEquals(4.0, a.getGrad()[0], 1e-9);
        // d(a*b)/db = a = 3.0
        assertEquals(3.0, b.getGrad()[0], 1e-9);
        // output grad should be seeded to 1.0
        assertEquals(1.0, output.getGrad()[0], 1e-9);
    }

    // --- 6.4: Gradient accumulation ---

    /**
     * One tensor used as input to two operations:
     * a → op1 → mid1 (= a * 2)
     * a → op2 → mid2 (= a * 3)
     * mid1 + mid2 → op3 → output (= a*2 + a*3 = 5a)
     * d(output)/da = 2 + 3 = 5
     */
    @Test
    void gradientAccumulationFromMultipleOperations() {
        Tensor a = new Tensor(new double[]{2.0}, new int[]{1}, true);

        // op1: mid1 = a * 2
        Tensor mid1 = new Tensor(new double[]{a.getData()[0] * 2.0}, new int[]{1}, true);
        new ScalarScaleOp(a, mid1, 2.0);

        // op2: mid2 = a * 3
        Tensor mid2 = new Tensor(new double[]{a.getData()[0] * 3.0}, new int[]{1}, true);
        new ScalarScaleOp(a, mid2, 3.0);

        // op3: output = mid1 + mid2
        Tensor output = new Tensor(
                new double[]{mid1.getData()[0] + mid2.getData()[0]}, new int[]{1}, true);
        new ScalarAddOp(mid1, mid2, output);

        output.backward();

        // a is used in both op1 and op2, so grad should be 2 + 3 = 5
        assertEquals(5.0, a.getGrad()[0], 1e-9);
    }

    // --- 6.5: backward() on non-scalar throws ---

    @Test
    void backwardOnNonScalarThrows() {
        Tensor t = new Tensor(new double[]{1, 2, 3}, new int[]{3}, true);
        assertThrows(IllegalStateException.class, t::backward);
    }

    // ========== Test-only Operation implementations ==========

    /**
     * Test-only operation: output = a * b (element-wise for scalars).
     * Gradients: d/da = b, d/db = a
     */
    private static class ScalarMultiplyOp extends Operation {
        ScalarMultiplyOp(Tensor a, Tensor b, Tensor output) {
            super(new Tensor[]{a, b}, output);
        }

        @Override
        public void backward() {
            Tensor a = inputs[0];
            Tensor b = inputs[1];
            // Accumulate: d(output)/da = b.data, d(output)/db = a.data
            if (a.isRequiresGrad()) {
                a.getGrad()[0] += output.getGrad()[0] * b.getData()[0];
            }
            if (b.isRequiresGrad()) {
                b.getGrad()[0] += output.getGrad()[0] * a.getData()[0];
            }
        }
    }

    /**
     * Test-only operation: output = input * scale (scalar constant).
     * Gradient: d/d(input) = scale
     */
    private static class ScalarScaleOp extends Operation {
        private final double scale;

        ScalarScaleOp(Tensor input, Tensor output, double scale) {
            super(new Tensor[]{input}, output);
            this.scale = scale;
        }

        @Override
        public void backward() {
            Tensor input = inputs[0];
            if (input.isRequiresGrad()) {
                input.getGrad()[0] += output.getGrad()[0] * scale;
            }
        }
    }

    /**
     * Test-only operation: output = a + b (scalar addition).
     * Gradients: d/da = 1, d/db = 1
     */
    private static class ScalarAddOp extends Operation {
        ScalarAddOp(Tensor a, Tensor b, Tensor output) {
            super(new Tensor[]{a, b}, output);
        }

        @Override
        public void backward() {
            Tensor a = inputs[0];
            Tensor b = inputs[1];
            if (a.isRequiresGrad()) {
                a.getGrad()[0] += output.getGrad()[0];
            }
            if (b.isRequiresGrad()) {
                b.getGrad()[0] += output.getGrad()[0];
            }
        }
    }
}
