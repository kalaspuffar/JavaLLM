package org.ea.javallm.model;

import org.ea.javallm.autograd.Tensor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for EncoderDecoderModel: output shape with different src/tgt vocab sizes,
 * named parameters with encoder/decoder prefixes, and forward+backward smoke test.
 */
class EncoderDecoderModelTest {

    private Random rng;

    @BeforeEach
    void setUp() {
        rng = new Random(42);
    }

    @Test
    void forwardProducesCorrectLogitShape() {
        EncoderDecoderModel model = new EncoderDecoderModel(
                30, 35, 16, 1, 4, 64, 32, rng);

        int[][] src = {{1, 5, 3, 7, 2, 4, 8, 9}};
        int[][] tgt = {{2, 6, 4, 1, 7}};

        Tensor logits = model.forward(src, tgt);

        assertArrayEquals(new int[]{1, 5, 35}, logits.getShape());
    }

    @Test
    void differentSrcAndTgtVocabSizes() {
        EncoderDecoderModel model = new EncoderDecoderModel(
                30, 35, 16, 1, 4, 64, 32, rng);

        int[][] src = {{1, 2, 3}};
        int[][] tgt = {{4, 5}};

        Tensor logits = model.forward(src, tgt);

        // Output should have tgtVocabSize=35, not srcVocabSize=30
        assertEquals(35, logits.size(2));
    }

    @Test
    void batchedForwardShape() {
        EncoderDecoderModel model = new EncoderDecoderModel(
                20, 25, 16, 1, 4, 64, 32, rng);

        int[][] src = {{1, 2, 3, 4}, {5, 6, 7, 8}};
        int[][] tgt = {{9, 10, 11}, {12, 13, 14}};

        Tensor logits = model.forward(src, tgt);

        assertArrayEquals(new int[]{2, 3, 25}, logits.getShape());
    }

    @Test
    void namedParametersHaveEncoderDecoderPrefixes() {
        EncoderDecoderModel model = new EncoderDecoderModel(
                20, 25, 16, 2, 4, 64, 32, rng);

        Map<String, Tensor> named = model.getNamedParameters();

        assertTrue(named.containsKey("srcEmbedding.weight"));
        assertTrue(named.containsKey("tgtEmbedding.weight"));
        assertTrue(named.containsKey("encoder.block.0.selfAttn.W_Q.weight"));
        assertTrue(named.containsKey("encoder.block.1.selfAttn.W_Q.weight"));
        assertTrue(named.containsKey("decoder.block.0.selfAttn.W_Q.weight"));
        assertTrue(named.containsKey("decoder.block.0.crossAttn.W_Q.weight"));
        assertTrue(named.containsKey("decoder.block.1.crossAttn.W_K.weight"));
        assertTrue(named.containsKey("finalNorm.gamma"));
        assertTrue(named.containsKey("outputProjection.weight"));
        assertTrue(named.containsKey("outputProjection.bias"));
    }

    @Test
    void namedParameterCountMatchesGetParameters() {
        EncoderDecoderModel model = new EncoderDecoderModel(
                20, 25, 16, 1, 4, 64, 32, rng);

        Map<String, Tensor> named = model.getNamedParameters();
        List<Tensor> params = model.getParameters();

        assertEquals(params.size(), named.size(),
                "Named parameter count should match getParameters() count");
    }

    @Test
    void forwardBackwardSmokeTest() {
        EncoderDecoderModel model = new EncoderDecoderModel(
                10, 12, 8, 1, 2, 32, 16, rng);

        int[][] src = {{1, 2, 3}};
        int[][] tgt = {{4, 5}};

        Tensor logits = model.forward(src, tgt);

        // Compute a simple scalar loss and run backward
        Tensor targets = new Tensor(new double[]{7, 3}, new int[]{2}, false);
        Tensor logitsFlat = logits.reshape(2, 12);
        Tensor loss = logitsFlat.crossEntropy(targets);

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
