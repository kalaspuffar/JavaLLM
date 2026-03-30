package org.ea.javallm.layers;

import org.ea.javallm.autograd.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for PositionalEncoding: verifies PE values at known positions
 * match the sinusoidal formula, and forward adds encoding to input.
 */
class PositionalEncodingTest {

    @Test
    void encodingValuesAtPositionZero() {
        PositionalEncoding pe = new PositionalEncoding(10, 4);
        Tensor table = pe.getEncodingTable();

        // PE(0, 0) = sin(0 / 10000^(0/4)) = sin(0) = 0.0
        assertEquals(0.0, table.get(0, 0), 1e-9, "PE(0,0) should be sin(0) = 0");

        // PE(0, 1) = cos(0 / 10000^(0/4)) = cos(0) = 1.0
        assertEquals(1.0, table.get(0, 1), 1e-9, "PE(0,1) should be cos(0) = 1");

        // PE(0, 2) = sin(0 / 10000^(2/4)) = sin(0) = 0.0
        assertEquals(0.0, table.get(0, 2), 1e-9, "PE(0,2) should be sin(0) = 0");

        // PE(0, 3) = cos(0 / 10000^(2/4)) = cos(0) = 1.0
        assertEquals(1.0, table.get(0, 3), 1e-9, "PE(0,3) should be cos(0) = 1");
    }

    @Test
    void encodingValuesAtPositionOne() {
        PositionalEncoding pe = new PositionalEncoding(10, 4);
        Tensor table = pe.getEncodingTable();

        // PE(1, 0) = sin(1 / 10000^(0/4)) = sin(1)
        assertEquals(Math.sin(1.0), table.get(1, 0), 1e-9);

        // PE(1, 1) = cos(1 / 10000^(0/4)) = cos(1)
        assertEquals(Math.cos(1.0), table.get(1, 1), 1e-9);

        // PE(1, 2) = sin(1 / 10000^(2/4)) = sin(1/100)
        assertEquals(Math.sin(1.0 / 100.0), table.get(1, 2), 1e-9);

        // PE(1, 3) = cos(1 / 10000^(2/4)) = cos(1/100)
        assertEquals(Math.cos(1.0 / 100.0), table.get(1, 3), 1e-9);
    }

    @Test
    void forwardAddsEncodingToInput() {
        PositionalEncoding pe = new PositionalEncoding(10, 4);
        Tensor table = pe.getEncodingTable();

        // Create a zero input — output should equal the encoding values
        Tensor input = Tensor.zeros(2, 5, 4);
        Tensor output = pe.forward(input);

        assertArrayEquals(new int[]{2, 5, 4}, output.getShape());

        // Each batch element should have the same PE values added
        for (int b = 0; b < 2; b++) {
            for (int pos = 0; pos < 5; pos++) {
                for (int d = 0; d < 4; d++) {
                    assertEquals(table.get(pos, d), output.get(b, pos, d), 1e-9,
                            "PE should be added at batch=" + b + " pos=" + pos + " dim=" + d);
                }
            }
        }
    }

    @Test
    void forwardOutputShapeMatchesInput() {
        PositionalEncoding pe = new PositionalEncoding(100, 8);
        Tensor input = Tensor.zeros(3, 20, 8);

        Tensor output = pe.forward(input);

        assertArrayEquals(new int[]{3, 20, 8}, output.getShape());
    }

    @Test
    void noLearnableParameters() {
        PositionalEncoding pe = new PositionalEncoding(10, 4);
        assertTrue(pe.getParameters().isEmpty());
    }
}
