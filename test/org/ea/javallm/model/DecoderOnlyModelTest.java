package org.ea.javallm.model;

import org.ea.javallm.autograd.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for DecoderOnlyModel: output shape, weight tying, named parameters,
 * and forward+backward smoke test.
 */
class DecoderOnlyModelTest {

    private Random rng;

    @BeforeEach
    void setUp() {
        rng = new Random(42);
    }

    @Test
    void forwardProducesCorrectLogitShape() {
        DecoderOnlyModel model = new DecoderOnlyModel(
                65, 32, 2, 4, 128, 64, rng);

        int[][] tokenIds = {{1, 5, 10, 3, 7}, {2, 8, 4, 6, 1}};
        Tensor logits = model.forward(tokenIds);

        assertArrayEquals(new int[]{2, 5, 65}, logits.getShape());
    }

    @Test
    void getParametersHasNoDuplicates() {
        DecoderOnlyModel model = new DecoderOnlyModel(
                30, 16, 2, 4, 64, 32, rng);

        List<Tensor> params = model.getParameters();

        // Use identity-based set to detect duplicate Tensor references
        Set<Tensor> uniqueParams = new HashSet<>();
        for (Tensor p : params) {
            assertTrue(uniqueParams.add(p),
                    "getParameters() should not contain duplicate Tensor references");
        }
    }

    @Test
    void getNamedParametersHasUniqueKeys() {
        DecoderOnlyModel model = new DecoderOnlyModel(
                30, 16, 2, 4, 64, 32, rng);

        Map<String, Tensor> named = model.getNamedParameters();

        // Map keys are inherently unique, but verify the count matches
        assertEquals(named.size(), model.getParameters().size(),
                "Named parameters count should match getParameters() count");

        // Verify some expected names exist
        assertTrue(named.containsKey("embedding.weight"));
        assertTrue(named.containsKey("finalNorm.gamma"));
        assertTrue(named.containsKey("finalNorm.beta"));
        assertTrue(named.containsKey("block.0.selfAttn.W_Q.weight"));
        assertTrue(named.containsKey("block.1.ffn.linear1.weight"));
    }

    @Test
    void weightTyingEmbeddingAppearsOnce() {
        DecoderOnlyModel model = new DecoderOnlyModel(
                30, 16, 1, 4, 64, 32, rng);

        Map<String, Tensor> named = model.getNamedParameters();
        Tensor embeddingWeight = named.get("embedding.weight");
        assertNotNull(embeddingWeight);

        // The embedding weight should appear exactly once in the named parameters
        long count = named.values().stream()
                .filter(t -> t == embeddingWeight)
                .count();
        assertEquals(1, count, "Embedding weight should appear exactly once (weight tying)");
    }

    @Test
    void forwardBackwardSmokeTest() {
        DecoderOnlyModel model = new DecoderOnlyModel(
                10, 8, 1, 2, 32, 16, rng);

        int[][] tokenIds = {{1, 2, 3}};
        Tensor logits = model.forward(tokenIds);

        // Compute a simple scalar loss (sum of first logit) and run backward
        Tensor targets = new Tensor(new double[]{2, 5, 1}, new int[]{3}, false);
        Tensor logitsFlat = logits.reshape(3, 10);
        Tensor loss = logitsFlat.crossEntropy(targets);

        // This should run without errors
        assertDoesNotThrow(loss::backward);

        // Verify some gradients are non-zero
        List<Tensor> params = model.getParameters();
        boolean anyNonZeroGrad = false;
        for (Tensor p : params) {
            if (p.getGrad() != null) {
                for (double g : p.getGrad()) {
                    if (Math.abs(g) > 1e-12) {
                        anyNonZeroGrad = true;
                        break;
                    }
                }
            }
            if (anyNonZeroGrad) break;
        }
        assertTrue(anyNonZeroGrad, "At least some parameters should have non-zero gradients");
    }
}
