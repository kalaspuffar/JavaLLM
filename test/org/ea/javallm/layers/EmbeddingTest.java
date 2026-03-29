package org.ea.javallm.layers;

import org.ea.javallm.autograd.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the Embedding layer: output shape for 1D and 2D indices,
 * and weight accessibility for weight tying.
 */
class EmbeddingTest {

    private Random rng;

    @BeforeEach
    void setUp() {
        rng = new Random(42);
    }

    @Test
    void outputShapeCorrectFor1DIndices() {
        Embedding embedding = new Embedding(10, 4, rng);
        Tensor indices = new Tensor(new double[]{2, 5, 7}, new int[]{3}, false);

        Tensor output = embedding.forward(indices);

        assertArrayEquals(new int[]{3, 4}, output.getShape());
    }

    @Test
    void outputShapeCorrectFor2DIndices() {
        Embedding embedding = new Embedding(10, 4, rng);
        Tensor indices = new Tensor(new double[]{2, 5, 7, 1, 3, 9}, new int[]{2, 3}, false);

        Tensor output = embedding.forward(indices);

        assertArrayEquals(new int[]{2, 3, 4}, output.getShape());
    }

    @Test
    void weightIsAccessibleForTying() {
        Embedding embedding = new Embedding(10, 4, rng);

        // The weight field should be the same tensor used for lookups
        assertNotNull(embedding.weight);
        assertArrayEquals(new int[]{10, 4}, embedding.weight.getShape());
        assertTrue(embedding.weight.isRequiresGrad());
    }

    @Test
    void forwardReturnsCorrectEmbeddingValues() {
        Embedding embedding = new Embedding(10, 4, rng);
        Tensor indices = new Tensor(new double[]{0, 2}, new int[]{2}, false);

        Tensor output = embedding.forward(indices);

        // Row 0 of output should match row 0 of weight
        for (int j = 0; j < 4; j++) {
            assertEquals(embedding.weight.get(0, j), output.get(0, j), 1e-9);
            assertEquals(embedding.weight.get(2, j), output.get(1, j), 1e-9);
        }
    }

    @Test
    void getParametersReturnsSingleWeight() {
        Embedding embedding = new Embedding(10, 4, rng);
        assertEquals(1, embedding.getParameters().size());
        assertSame(embedding.weight, embedding.getParameters().get(0));
    }
}
