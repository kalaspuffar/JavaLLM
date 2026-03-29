package org.ea.javallm.layers;

import org.ea.javallm.autograd.Tensor;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for LayerNorm: verifies output has zero mean and unit variance
 * with default gamma/beta, and parameters are accessible.
 */
class LayerNormTest {

    @Test
    void outputHasZeroMeanAndUnitVariance() {
        LayerNorm norm = new LayerNorm(4);
        Random rng = new Random(42);

        // Create a (2, 3, 4) input with non-trivial values
        double[] data = new double[24];
        for (int i = 0; i < 24; i++) {
            data[i] = rng.nextGaussian() * 5.0 + 3.0;
        }
        Tensor input = new Tensor(data, new int[]{2, 3, 4}, false);

        Tensor output = norm.forward(input);

        assertArrayEquals(new int[]{2, 3, 4}, output.getShape());

        // Each 4-element slice along the last dimension should have mean ≈ 0 and variance ≈ 1
        for (int b = 0; b < 2; b++) {
            for (int s = 0; s < 3; s++) {
                double mean = 0.0;
                for (int d = 0; d < 4; d++) {
                    mean += output.get(b, s, d);
                }
                mean /= 4.0;
                assertEquals(0.0, mean, 1e-5,
                        "Mean should be ≈ 0 at batch=" + b + " seq=" + s);

                double variance = 0.0;
                for (int d = 0; d < 4; d++) {
                    double diff = output.get(b, s, d) - mean;
                    variance += diff * diff;
                }
                variance /= 4.0;
                assertEquals(1.0, variance, 1e-4,
                        "Variance should be ≈ 1 at batch=" + b + " seq=" + s);
            }
        }
    }

    @Test
    void getParametersReturnsGammaAndBeta() {
        LayerNorm norm = new LayerNorm(8);
        assertEquals(2, norm.getParameters().size());
        assertSame(norm.getGamma(), norm.getParameters().get(0));
        assertSame(norm.getBeta(), norm.getParameters().get(1));
    }

    @Test
    void gammaInitializedToOnes() {
        LayerNorm norm = new LayerNorm(4);
        for (double v : norm.getGamma().getData()) {
            assertEquals(1.0, v);
        }
    }

    @Test
    void betaInitializedToZeros() {
        LayerNorm norm = new LayerNorm(4);
        for (double v : norm.getBeta().getData()) {
            assertEquals(0.0, v);
        }
    }

    @Test
    void parametersHaveRequiresGrad() {
        LayerNorm norm = new LayerNorm(4);
        assertTrue(norm.getGamma().isRequiresGrad());
        assertTrue(norm.getBeta().isRequiresGrad());
    }
}
