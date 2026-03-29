package org.ea.javallm.data;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class ReversalTaskGeneratorTest {

    @Test
    void reversalIsCorrect() {
        CharTokenizer tok = CharTokenizer.fromText("abc", true);
        // Fixed seed with minLen=maxLen=3 ensures we get a known-length string
        ReversalTaskGenerator gen = new ReversalTaskGenerator(tok, 3, 3, new Random(42));
        ReversalTaskGenerator.ReversalBatch batch = gen.generateBatch(1);

        int[][] src = batch.getSource();
        int[][] tgtInput = batch.getTgtInput();
        int[][] tgtTarget = batch.getTgtTarget();

        // Source is 3 characters, so 3 tokens
        assertEquals(3, src[0].length);

        // Decoder input starts with SOS
        assertEquals(CharTokenizer.SOS, tgtInput[0][0]);

        // Decoder target ends with EOS at position = string length
        assertEquals(CharTokenizer.EOS, tgtTarget[0][3]);

        // The reversal content in tgtInput[1..3] should match tgtTarget[0..2]
        for (int i = 0; i < 3; i++) {
            assertEquals(tgtInput[0][i + 1], tgtTarget[0][i],
                    "Decoder input (after SOS) and target (before EOS) should match");
        }

        // Verify actual reversal: decode source and reversed content, compare
        String srcStr = tok.decode(src[0]);
        int[] reversedTokens = new int[3];
        System.arraycopy(tgtTarget[0], 0, reversedTokens, 0, 3);
        String revStr = tok.decode(reversedTokens);
        assertEquals(new StringBuilder(srcStr).reverse().toString(), revStr);
    }

    @Test
    void sosAndEosArePresent() {
        CharTokenizer tok = CharTokenizer.fromText("abcdef", true);
        ReversalTaskGenerator gen = new ReversalTaskGenerator(tok, 2, 2, new Random(0));
        ReversalTaskGenerator.ReversalBatch batch = gen.generateBatch(4);

        for (int i = 0; i < 4; i++) {
            assertEquals(CharTokenizer.SOS, batch.getTgtInput()[i][0],
                    "Every decoder input should start with SOS");
            assertEquals(CharTokenizer.EOS, batch.getTgtTarget()[i][2],
                    "Every decoder target should end with EOS at length position");
        }
    }

    @Test
    void paddingIsAppliedForVariableLengths() {
        CharTokenizer tok = CharTokenizer.fromText("abcdef", true);
        // minLen=1, maxLen=4 will produce strings of varying lengths
        ReversalTaskGenerator gen = new ReversalTaskGenerator(tok, 1, 4, new Random(99));
        ReversalTaskGenerator.ReversalBatch batch = gen.generateBatch(8);

        int[][] src = batch.getSource();
        int[][] tgtInput = batch.getTgtInput();
        int[][] tgtTarget = batch.getTgtTarget();

        // All rows in each array should have the same length (padded to max)
        int srcLen = src[0].length;
        int tgtLen = tgtInput[0].length;
        for (int i = 0; i < 8; i++) {
            assertEquals(srcLen, src[i].length, "All source rows should have equal length");
            assertEquals(tgtLen, tgtInput[i].length, "All tgtInput rows should have equal length");
            assertEquals(tgtLen, tgtTarget[i].length, "All tgtTarget rows should have equal length");
        }
    }

    @Test
    void requiresSpecialTokens() {
        CharTokenizer tokNoSpecial = CharTokenizer.fromText("abc");
        assertThrows(IllegalArgumentException.class,
                () -> new ReversalTaskGenerator(tokNoSpecial, 1, 3, new Random(0)));
    }
}
