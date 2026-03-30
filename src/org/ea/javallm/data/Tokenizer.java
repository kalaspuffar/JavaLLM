package org.ea.javallm.data;

/**
 * Common interface for all tokenizers.
 *
 * A tokenizer converts between human-readable text and integer token IDs
 * that the model operates on. The granularity of a "token" is defined by the
 * implementation — it could be a single character ({@link CharTokenizer}) or
 * a whole word ({@link WordTokenizer}).
 *
 * When special tokens are enabled (for encoder-decoder tasks), IDs 0–2 are
 * reserved for PAD, SOS, and EOS. Content token IDs start at 3.
 */
public interface Tokenizer {

    int PAD = 0;
    int SOS = 1;
    int EOS = 2;

    /**
     * Encodes a string into an array of integer token IDs.
     *
     * @throws IllegalArgumentException if the string contains a token not in the vocabulary
     */
    int[] encode(String text);

    /**
     * Decodes an array of integer token IDs back into a string.
     * Special tokens (PAD, SOS, EOS) are skipped during decoding.
     */
    String decode(int[] ids);

    /**
     * Returns the total vocabulary size, including special tokens if enabled.
     */
    int getVocabSize();

    /**
     * Returns whether this tokenizer includes special tokens (PAD, SOS, EOS).
     */
    boolean hasSpecialTokens();
}
