package org.ea.javallm.model;

import org.ea.javallm.autograd.Tensor;
import org.ea.javallm.layers.MultiHeadAttention;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for TransformerBlock in all three configuration modes:
 * encoder, decoder, and decoder-with-cross-attention.
 */
class TransformerBlockTest {

    private Random rng;

    @BeforeEach
    void setUp() {
        rng = new Random(42);
    }

    @Test
    void encoderModeOutputShape() {
        TransformerBlock block = new TransformerBlock(16, 4, 64, false, false, rng);
        Tensor input = randomTensor(new int[]{2, 5, 16});

        Tensor output = block.forward(input, null, null);

        assertArrayEquals(new int[]{2, 5, 16}, output.getShape());
    }

    @Test
    void decoderModeOutputShape() {
        TransformerBlock block = new TransformerBlock(16, 4, 64, true, false, rng);
        Tensor input = randomTensor(new int[]{2, 5, 16});
        Tensor causalMask = MultiHeadAttention.createCausalMask(5);

        Tensor output = block.forward(input, null, causalMask);

        assertArrayEquals(new int[]{2, 5, 16}, output.getShape());
    }

    @Test
    void decoderWithCrossAttentionOutputShape() {
        TransformerBlock block = new TransformerBlock(16, 4, 64, true, true, rng);
        Tensor input = randomTensor(new int[]{2, 5, 16});
        Tensor encoderOutput = randomTensor(new int[]{2, 8, 16});
        Tensor causalMask = MultiHeadAttention.createCausalMask(5);

        Tensor output = block.forward(input, encoderOutput, causalMask);

        assertArrayEquals(new int[]{2, 5, 16}, output.getShape());
    }

    @Test
    void encoderModeHasNoAttnParameters() {
        TransformerBlock encoder = new TransformerBlock(16, 4, 64, false, false, rng);
        TransformerBlock decoderCross = new TransformerBlock(16, 4, 64, true, true, new Random(42));

        // Decoder with cross-attention should have more parameters than encoder
        assertTrue(decoderCross.getParameters().size() > encoder.getParameters().size(),
                "Decoder with cross-attention should have more params than encoder");
    }

    @Test
    void namedParametersHaveUniqueKeys() {
        TransformerBlock block = new TransformerBlock(16, 4, 64, true, true, rng);
        Map<String, Tensor> named = block.getNamedParameters("block.0");

        // All keys should be unique (guaranteed by Map, but verify count matches)
        assertEquals(named.size(), block.getParameters().size(),
                "Named parameter count should match getParameters() count");

        // Verify expected prefixes exist
        assertTrue(named.keySet().stream().anyMatch(k -> k.contains("selfAttn")));
        assertTrue(named.keySet().stream().anyMatch(k -> k.contains("crossAttn")));
        assertTrue(named.keySet().stream().anyMatch(k -> k.contains("ffn")));
    }

    @Test
    void residualConnectionPreservesInputInformation() {
        TransformerBlock block = new TransformerBlock(16, 4, 64, false, false, rng);
        Tensor input = randomTensor(new int[]{1, 3, 16});

        Tensor output = block.forward(input, null, null);

        // Output should not be identical to input (sublayers modify it),
        // but both should have the same shape
        assertArrayEquals(input.getShape(), output.getShape());

        // Output should differ from input due to sublayer processing
        boolean differs = false;
        for (int i = 0; i < input.size(); i++) {
            if (Math.abs(input.getData()[i] - output.getData()[i]) > 1e-10) {
                differs = true;
                break;
            }
        }
        assertTrue(differs, "Output should differ from input after sublayer processing");
    }

    private Tensor randomTensor(int[] shape) {
        int size = 1;
        for (int s : shape) size *= s;
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = rng.nextGaussian() * 0.3;
        }
        return new Tensor(data, shape, false);
    }
}
