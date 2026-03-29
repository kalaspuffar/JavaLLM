package org.ea.javallm.data;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class SequenceBatcherTest {

    @Test
    void inputAndTargetAreOffsetByOne() {
        int[] tokens = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        int contextLen = 3;
        int batchSize = 1;
        // Use a fixed seed so we get deterministic starting positions
        SequenceBatcher batcher = new SequenceBatcher(tokens, contextLen, batchSize, new Random(42));

        int[][] input = batcher.nextInputBatch();
        int[][] target = batcher.nextTargetBatch();

        assertEquals(1, input.length);
        assertEquals(contextLen, input[0].length);
        assertEquals(contextLen, target[0].length);

        // For any starting position pos, target[i] == input[i] + 1
        // because target = tokens[pos+1 : pos+contextLen+1]
        for (int i = 0; i < contextLen; i++) {
            assertEquals(input[0][i] + 1, target[0][i],
                    "Target should be input shifted by one at position " + i);
        }
    }

    @Test
    void batchDimensionsAreCorrect() {
        int[] tokens = new int[100];
        for (int i = 0; i < tokens.length; i++) tokens[i] = i;

        int contextLen = 8;
        int batchSize = 4;
        SequenceBatcher batcher = new SequenceBatcher(tokens, contextLen, batchSize, new Random(7));

        int[][] input = batcher.nextInputBatch();
        int[][] target = batcher.nextTargetBatch();

        assertEquals(batchSize, input.length);
        assertEquals(batchSize, target.length);
        for (int b = 0; b < batchSize; b++) {
            assertEquals(contextLen, input[b].length);
            assertEquals(contextLen, target[b].length);
        }
    }

    @Test
    void resetReshufflesPositions() {
        int[] tokens = new int[50];
        for (int i = 0; i < tokens.length; i++) tokens[i] = i;

        SequenceBatcher batcher = new SequenceBatcher(tokens, 3, 1, new Random(42));
        int[][] firstEpochBatch = batcher.nextInputBatch();
        int firstPos = firstEpochBatch[0][0];

        batcher.reset();
        int[][] secondEpochBatch = batcher.nextInputBatch();
        int secondPos = secondEpochBatch[0][0];

        // With different random shuffles, the first position should differ
        // (extremely unlikely to be the same with 46 possible positions)
        assertNotEquals(firstPos, secondPos,
                "After reset, positions should be reshuffled");
    }

    @Test
    void hasNextReturnsFalseWhenExhausted() {
        // tokens.length=4, contextLen=2, so maxStart = 4-2-1 = 1
        // Valid positions: {0, 1} → 2 positions, batchSize=2 → exactly one batch
        int[] tokens = {0, 1, 2, 3};
        SequenceBatcher batcher = new SequenceBatcher(tokens, 2, 2, new Random(0));
        assertTrue(batcher.hasNext());
        batcher.nextInputBatch();
        assertFalse(batcher.hasNext());
    }
}
