package org.ea.javallm.trainers;

import org.ea.javallm.autograd.Tensor;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.*;

class AdamOptimizerTest {

    @Test
    void minimizesXSquaredTowardZero() {
        // Create a single scalar parameter initialized to 5.0
        Tensor x = new Tensor(new double[]{5.0}, new int[]{1}, true);
        AdamOptimizer optimizer = new AdamOptimizer(Collections.singletonList(x), 0.1);

        // Manually minimize f(x) = x^2 by setting grad = 2x at each step
        for (int step = 0; step < 1000; step++) {
            optimizer.zeroGrad();
            // Gradient of x^2 is 2x
            x.getGrad()[0] = 2.0 * x.getData()[0];
            optimizer.step();
        }

        // After 1000 steps, x should be very close to 0
        assertEquals(0.0, x.getData()[0], 0.01,
                "Adam should minimize x^2 to near zero");
    }

    @Test
    void zeroGradClearsAllGradients() {
        Tensor a = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, true);
        Tensor b = new Tensor(new double[]{3.0, 4.0, 5.0}, new int[]{3}, true);
        AdamOptimizer optimizer = new AdamOptimizer(Arrays.asList(a, b));

        // Simulate some gradient values
        a.getGrad()[0] = 10.0;
        a.getGrad()[1] = 20.0;
        b.getGrad()[0] = 30.0;
        b.getGrad()[1] = 40.0;
        b.getGrad()[2] = 50.0;

        optimizer.zeroGrad();

        for (double g : a.getGrad()) {
            assertEquals(0.0, g, "Gradient should be zeroed");
        }
        for (double g : b.getGrad()) {
            assertEquals(0.0, g, "Gradient should be zeroed");
        }
    }

    @Test
    void stepOnlyReadsGradAndUpdatesData() {
        // Verify that step() modifies data based on grad, without side effects
        Tensor param = new Tensor(new double[]{3.0}, new int[]{1}, true);
        AdamOptimizer optimizer = new AdamOptimizer(Collections.singletonList(param), 0.01);

        double before = param.getData()[0];
        param.getGrad()[0] = 1.0;
        optimizer.step();
        double after = param.getData()[0];

        assertNotEquals(before, after,
                "step() should update parameter data when gradient is nonzero");
        assertTrue(after < before,
                "With positive gradient, parameter should decrease");
    }
}
