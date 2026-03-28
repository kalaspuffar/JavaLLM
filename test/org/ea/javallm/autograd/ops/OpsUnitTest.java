package org.ea.javallm.autograd.ops;

import org.ea.javallm.autograd.Tensor;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for specific operation behaviours: numerical stability,
 * output shapes, error conditions, and sparse gradient correctness.
 */
class OpsUnitTest {

    // --- Softmax numerical stability ---

    @Test
    void softmaxWithLargeValuesDoesNotProduceNaN() {
        double[] data = new double[]{1000.0, 999.0, 998.0, 997.0, 996.0};
        Tensor t = new Tensor(data, new int[]{1, 5}, false);
        Tensor result = t.softmax(1);

        double[] resultData = result.getData();
        for (double v : resultData) {
            assertFalse(Double.isNaN(v), "Softmax produced NaN");
            assertFalse(Double.isInfinite(v), "Softmax produced Infinity");
        }

        // Verify row sums to 1
        double sum = 0.0;
        for (double v : resultData) sum += v;
        assertEquals(1.0, sum, 1e-6);
    }

    @Test
    void softmaxRowsSumToOne() {
        Tensor t = new Tensor(new double[]{
                1.0, 2.0, 3.0, 4.0, 5.0,
                5.0, 4.0, 3.0, 2.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0
        }, new int[]{3, 5}, false);

        Tensor result = t.softmax(1);

        for (int row = 0; row < 3; row++) {
            double sum = 0.0;
            for (int col = 0; col < 5; col++) {
                sum += result.get(row, col);
            }
            assertEquals(1.0, sum, 1e-6, "Row " + row + " should sum to 1.0");
        }
    }

    // --- CrossEntropy scalar output ---

    @Test
    void crossEntropyReturnsScalar() {
        Tensor logits = new Tensor(new double[]{
                1.0, 2.0, 3.0, 4.0, 5.0,
                5.0, 4.0, 3.0, 2.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0
        }, new int[]{3, 5}, true);
        Tensor targets = new Tensor(new double[]{4, 0, 2}, new int[]{3}, false);

        Tensor loss = logits.crossEntropy(targets);

        assertEquals(1, loss.size(), "CrossEntropy output should be scalar");
        assertTrue(loss.getData()[0] > 0, "Loss should be positive");
    }

    // --- Reshape invalid shape rejection ---

    @Test
    void reshapeRejectsInvalidShape() {
        Tensor t = new Tensor(new double[12], new int[]{2, 6}, false);
        assertThrows(IllegalArgumentException.class,
                () -> t.reshape(3, 3),
                "Should reject reshape from 12 elements to 9 elements");
    }

    @Test
    void reshapeValidPreservesData() {
        Tensor t = new Tensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                new int[]{2, 6}, false);
        Tensor reshaped = t.reshape(3, 4);
        assertArrayEquals(new int[]{3, 4}, reshaped.getShape());
        // Data order is preserved in flat layout
        assertEquals(1.0, reshaped.get(0, 0));
        assertEquals(5.0, reshaped.get(1, 0));
        assertEquals(12.0, reshaped.get(2, 3));
    }

    // --- EmbeddingLookup sparse gradient correctness ---

    @Test
    void embeddingLookupOnlySelectedRowsReceiveGradient() {
        double[] weightData = new double[40]; // 10 x 4
        for (int i = 0; i < 40; i++) weightData[i] = i * 0.1;
        Tensor weights = new Tensor(weightData, new int[]{10, 4}, true);

        Tensor indices = new Tensor(new double[]{0, 2}, new int[]{2}, false);
        Tensor output = weights.embeddingLookup(indices);

        // Create a scalar loss by summing the output
        double sum = 0;
        for (double v : output.getData()) sum += v;
        Tensor loss = new Tensor(new double[]{sum}, new int[]{1}, true);

        // Wire a simple sum operation for backward
        new TestSumOp(output, loss);

        loss.backward();

        double[] grad = weights.getGrad();

        // Rows 0 and 2 should have gradient = 1.0 for each element
        for (int j = 0; j < 4; j++) {
            assertEquals(1.0, grad[0 * 4 + j], 1e-9, "Row 0 should have gradient");
            assertEquals(1.0, grad[2 * 4 + j], 1e-9, "Row 2 should have gradient");
        }

        // All other rows should have zero gradient
        for (int row = 0; row < 10; row++) {
            if (row == 0 || row == 2) continue;
            for (int j = 0; j < 4; j++) {
                assertEquals(0.0, grad[row * 4 + j], 1e-9,
                        "Row " + row + " should have zero gradient");
            }
        }
    }

    @Test
    void embeddingLookupDuplicateIndicesAccumulateGradient() {
        double[] weightData = new double[20]; // 5 x 4
        Tensor weights = new Tensor(weightData, new int[]{5, 4}, true);

        // Index 1 appears twice — gradient should be accumulated
        Tensor indices = new Tensor(new double[]{1, 3, 1}, new int[]{3}, false);
        Tensor output = weights.embeddingLookup(indices);

        double sum = 0;
        for (double v : output.getData()) sum += v;
        Tensor loss = new Tensor(new double[]{sum}, new int[]{1}, true);
        new TestSumOp(output, loss);

        loss.backward();

        double[] grad = weights.getGrad();

        // Row 1 was selected twice, so each element should have gradient = 2.0
        for (int j = 0; j < 4; j++) {
            assertEquals(2.0, grad[1 * 4 + j], 1e-9, "Row 1 grad should be 2.0 (two lookups)");
        }
        // Row 3 selected once
        for (int j = 0; j < 4; j++) {
            assertEquals(1.0, grad[3 * 4 + j], 1e-9, "Row 3 grad should be 1.0");
        }
    }

    // --- Transpose correctness ---

    @Test
    void transpose2DSwapsRowsAndColumns() {
        Tensor t = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, false);
        Tensor transposed = t.transpose(0, 1);
        assertArrayEquals(new int[]{3, 2}, transposed.getShape());
        assertEquals(t.get(0, 1), transposed.get(1, 0));
        assertEquals(t.get(1, 2), transposed.get(2, 1));
    }

    // --- Mask correctness ---

    @Test
    void maskFillsMaskedPositions() {
        Tensor input = new Tensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                new int[]{3, 4}, false);
        double[] maskData = new double[]{1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0};
        Tensor mask = new Tensor(maskData, new int[]{3, 4}, false);

        Tensor result = input.mask(mask, -1e9);

        // Column 3 should be filled
        assertEquals(-1e9, result.get(0, 3));
        assertEquals(-1e9, result.get(1, 3));
        assertEquals(-1e9, result.get(2, 3));
        // Other columns should pass through
        assertEquals(1.0, result.get(0, 0));
        assertEquals(6.0, result.get(1, 1));
    }

    // --- ReLU correctness ---

    @Test
    void reluZeroesNegativeValues() {
        Tensor t = new Tensor(new double[]{-2, -1, 0, 1, 2}, new int[]{5}, false);
        Tensor result = t.relu();
        assertEquals(0.0, result.get(0));
        assertEquals(0.0, result.get(1));
        assertEquals(0.0, result.get(2));
        assertEquals(1.0, result.get(3));
        assertEquals(2.0, result.get(4));
    }

    /** Test-only sum operation for building test graphs. */
    private static class TestSumOp extends org.ea.javallm.autograd.Operation {
        TestSumOp(Tensor input, Tensor output) {
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
