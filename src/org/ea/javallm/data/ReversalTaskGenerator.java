package org.ea.javallm.data;

import java.util.Random;

/**
 * Generates synthetic string-reversal tasks for encoder-decoder training.
 *
 * Each example consists of a random string drawn from the tokenizer's character
 * vocabulary and its reversal. The generator produces three parallel arrays per
 * batch: source tokens, decoder input tokens (with SOS prefix for teacher forcing),
 * and decoder target tokens (with EOS suffix). Shorter sequences within a batch
 * are padded to match the longest.
 *
 * Requires a tokenizer built with special tokens enabled (PAD, SOS, EOS).
 */
public class ReversalTaskGenerator {

    private final CharTokenizer tokenizer;
    private final int minLen;
    private final int maxLen;
    private final Random rng;

    /**
     * @param tokenizer a CharTokenizer with special tokens enabled
     * @param minLen    minimum random string length (inclusive)
     * @param maxLen    maximum random string length (inclusive)
     * @param rng       random number generator
     * @throws IllegalArgumentException if the tokenizer does not have special tokens
     */
    public ReversalTaskGenerator(CharTokenizer tokenizer, int minLen, int maxLen, Random rng) {
        if (!tokenizer.hasSpecialTokens()) {
            throw new IllegalArgumentException(
                    "ReversalTaskGenerator requires a tokenizer with special tokens enabled");
        }
        if (minLen < 1 || maxLen < minLen) {
            throw new IllegalArgumentException(
                    "Invalid length range: minLen=" + minLen + ", maxLen=" + maxLen);
        }
        this.tokenizer = tokenizer;
        this.minLen = minLen;
        this.maxLen = maxLen;
        this.rng = rng;
    }

    /**
     * Generates a batch of reversal examples.
     *
     * @param batchSize number of examples to generate
     * @return a {@link ReversalBatch} containing source, decoder input, and decoder target arrays
     */
    public ReversalBatch generateBatch(int batchSize) {
        // Generate random strings and compute the maximum sequence length for padding
        String[] strings = new String[batchSize];
        int maxStringLen = 0;
        for (int i = 0; i < batchSize; i++) {
            int len = minLen + rng.nextInt(maxLen - minLen + 1);
            strings[i] = generateRandomString(len);
            maxStringLen = Math.max(maxStringLen, len);
        }

        // Source: padded to maxStringLen
        // Decoder input/target: padded to maxStringLen + 1 (SOS/EOS adds one token)
        int srcPadLen = maxStringLen;
        int tgtPadLen = maxStringLen + 1;

        int[][] sourceOut = new int[batchSize][srcPadLen];
        int[][] tgtInputOut = new int[batchSize][tgtPadLen];
        int[][] tgtTargetOut = new int[batchSize][tgtPadLen];

        for (int i = 0; i < batchSize; i++) {
            String original = strings[i];
            int[] srcTokens = tokenizer.encode(original);
            int[] revTokens = tokenizer.encode(reverseString(original));

            // Source: original tokens, padded with PAD
            System.arraycopy(srcTokens, 0, sourceOut[i], 0, srcTokens.length);
            for (int p = srcTokens.length; p < srcPadLen; p++) {
                sourceOut[i][p] = CharTokenizer.PAD;
            }

            // Decoder input: SOS + reversed tokens, padded with PAD
            tgtInputOut[i][0] = CharTokenizer.SOS;
            System.arraycopy(revTokens, 0, tgtInputOut[i], 1, revTokens.length);
            for (int p = revTokens.length + 1; p < tgtPadLen; p++) {
                tgtInputOut[i][p] = CharTokenizer.PAD;
            }

            // Decoder target: reversed tokens + EOS, padded with PAD
            System.arraycopy(revTokens, 0, tgtTargetOut[i], 0, revTokens.length);
            tgtTargetOut[i][revTokens.length] = CharTokenizer.EOS;
            for (int p = revTokens.length + 1; p < tgtPadLen; p++) {
                tgtTargetOut[i][p] = CharTokenizer.PAD;
            }
        }

        return new ReversalBatch(sourceOut, tgtInputOut, tgtTargetOut);
    }

    private String generateRandomString(int length) {
        // Build alphabet from the tokenizer's character range
        int offset = tokenizer.getCharacterIdOffset();
        int numChars = tokenizer.getVocabSize() - offset;
        StringBuilder sb = new StringBuilder(length);
        for (int i = 0; i < length; i++) {
            int charId = offset + rng.nextInt(numChars);
            sb.append(tokenizer.decode(new int[]{charId}));
        }
        return sb.toString();
    }

    private static String reverseString(String s) {
        return new StringBuilder(s).reverse().toString();
    }

    /**
     * Holds the three parallel arrays produced by a reversal batch generation.
     */
    public static class ReversalBatch {
        private final int[][] source;
        private final int[][] tgtInput;
        private final int[][] tgtTarget;

        public ReversalBatch(int[][] source, int[][] tgtInput, int[][] tgtTarget) {
            this.source = source;
            this.tgtInput = tgtInput;
            this.tgtTarget = tgtTarget;
        }

        public int[][] getSource() { return source; }
        public int[][] getTgtInput() { return tgtInput; }
        public int[][] getTgtTarget() { return tgtTarget; }
    }
}
