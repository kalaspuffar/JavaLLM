package org.ea.javallm.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

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

    /**
     * Saves the vocabulary to a plain-text file so the tokenizer can be
     * reconstructed without the original training data.
     *
     * @param path file path to write the vocab file to
     * @throws IOException if the file cannot be written
     */
    void saveVocab(String path) throws IOException;

    /**
     * Loads a tokenizer from a previously saved vocab file.
     *
     * Reads the header line to determine the tokenizer type ({@code char} or
     * {@code word}) and delegates to the appropriate implementation's factory.
     *
     * @param path path to the vocab file
     * @return a Tokenizer reconstructed from the file
     * @throws IOException if the file cannot be read or has an unrecognised format
     */
    static Tokenizer loadVocab(String path) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(Path.of(path))) {
            String header = reader.readLine();
            if (header == null || !header.startsWith("#type=")) {
                throw new IOException("Invalid vocab file: missing header line");
            }

            if (header.startsWith("#type=char")) {
                return CharTokenizer.fromVocabFile(path);
            } else if (header.startsWith("#type=word")) {
                return WordTokenizer.fromVocabFile(path);
            } else {
                String type = header.substring("#type=".length()).split("\\s")[0];
                throw new IOException("Unknown tokenizer type in vocab file: " + type);
            }
        }
    }

    /**
     * Derives the vocab file path from a model file path.
     *
     * Replaces the file extension with {@code .vocab}, or appends {@code .vocab}
     * if the model path has no extension.
     *
     * @param modelPath path to the model file
     * @return the corresponding vocab file path
     */
    static String vocabPathForModel(String modelPath) {
        int lastDot = modelPath.lastIndexOf('.');
        int lastSep = Math.max(modelPath.lastIndexOf('/'), modelPath.lastIndexOf('\\'));
        // Only treat as extension if the dot comes after the last path separator
        if (lastDot > lastSep && lastDot > 0) {
            return modelPath.substring(0, lastDot) + ".vocab";
        }
        return modelPath + ".vocab";
    }
}
