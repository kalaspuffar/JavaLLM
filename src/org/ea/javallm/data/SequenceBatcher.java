package org.ea.javallm.data;

import java.util.Random;

/**
 * Produces shifted (input, target) batch pairs for next-token prediction training.
 *
 * For each batch element, a random starting position is sampled from the token array.
 * The input slice is {@code tokens[pos : pos+contextLen]} and the target slice is
 * {@code tokens[pos+1 : pos+contextLen+1]}, giving the model one token of lookahead
 * for each input position.
 *
 * Starting positions are pre-generated and shuffled. Call {@link #reset()} between
 * epochs to reshuffle for variety.
 */
public class SequenceBatcher {

    private final int[] tokens;
    private final int contextLen;
    private final int batchSize;
    private final Random rng;

    private int[] startPositions;
    private int cursor;

    // Cached batch arrays, reused across calls to avoid repeated allocation
    private int[][] inputBatch;
    private int[][] targetBatch;

    /**
     * @param tokens     the full tokenized text as integer IDs
     * @param contextLen number of tokens per training example
     * @param batchSize  number of examples per batch
     * @param rng        random number generator for sampling positions
     */
    public SequenceBatcher(int[] tokens, int contextLen, int batchSize, Random rng) {
        if (tokens.length < contextLen + 1) {
            throw new IllegalArgumentException(
                    "Token array length (" + tokens.length +
                    ") must be at least contextLen + 1 (" + (contextLen + 1) + ")");
        }
        this.tokens = tokens;
        this.contextLen = contextLen;
        this.batchSize = batchSize;
        this.rng = rng;
        this.inputBatch = new int[batchSize][contextLen];
        this.targetBatch = new int[batchSize][contextLen];

        int maxStartPosition = tokens.length - contextLen - 1;
        this.startPositions = generateShuffledPositions(maxStartPosition + 1);
        this.cursor = 0;
    }

    /**
     * Returns true if there are enough remaining positions to fill a complete batch.
     */
    public boolean hasNext() {
        return cursor + batchSize <= startPositions.length;
    }

    /**
     * Returns the next input batch. Each row is a sequence of {@code contextLen} token IDs.
     * Must be called before {@link #nextTargetBatch()} — they advance in lockstep.
     */
    public int[][] nextInputBatch() {
        fillBatches();
        return inputBatch;
    }

    /**
     * Returns the target batch corresponding to the most recent input batch.
     * Each row is shifted one position forward from the corresponding input row.
     */
    public int[][] nextTargetBatch() {
        return targetBatch;
    }

    /**
     * Reshuffles starting positions for a new epoch and resets the cursor.
     */
    public void reset() {
        int maxStartPosition = tokens.length - contextLen - 1;
        this.startPositions = generateShuffledPositions(maxStartPosition + 1);
        this.cursor = 0;
    }

    private void fillBatches() {
        for (int b = 0; b < batchSize; b++) {
            int pos = startPositions[cursor + b];
            System.arraycopy(tokens, pos, inputBatch[b], 0, contextLen);
            System.arraycopy(tokens, pos + 1, targetBatch[b], 0, contextLen);
        }
        cursor += batchSize;
    }

    /**
     * Generates an array of integers [0, count) and shuffles them using Fisher-Yates.
     */
    private int[] generateShuffledPositions(int count) {
        int[] positions = new int[count];
        for (int i = 0; i < count; i++) {
            positions[i] = i;
        }
        for (int i = count - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int temp = positions[i];
            positions[i] = positions[j];
            positions[j] = temp;
        }
        return positions;
    }
}
