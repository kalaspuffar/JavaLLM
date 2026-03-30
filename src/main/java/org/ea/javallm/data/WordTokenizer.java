package org.ea.javallm.data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

/**
 * Word-level tokenizer that builds a vocabulary from whitespace-delimited words.
 *
 * Words are sorted lexicographically to ensure deterministic ID assignment
 * regardless of their order in the training text. When special tokens are
 * enabled (for encoder-decoder tasks), PAD=0, SOS=1, EOS=2 are reserved
 * and word IDs start at 3.
 *
 * This tokenizer demonstrates that the Transformer architecture is agnostic
 * to token granularity — the same model can operate on characters, words, or
 * any other text unit. Compared to {@link CharTokenizer}, a word tokenizer
 * produces shorter sequences but has a much larger vocabulary.
 */
public class WordTokenizer implements Tokenizer {

    private final String[] vocabulary;
    private final Map<String, Integer> wordToId;
    private final int vocabSize;
    private final boolean includeSpecialTokens;
    private final int wordIdOffset;

    private WordTokenizer(String text, boolean includeSpecialTokens) {
        this.includeSpecialTokens = includeSpecialTokens;
        this.wordIdOffset = includeSpecialTokens ? 3 : 0;

        // Split on whitespace and collect unique words in sorted order
        TreeSet<String> uniqueWords = new TreeSet<>();
        for (String word : text.split("\\s+")) {
            if (!word.isEmpty()) {
                uniqueWords.add(word);
            }
        }

        this.vocabulary = uniqueWords.toArray(new String[0]);
        this.wordToId = new HashMap<>();
        for (int i = 0; i < vocabulary.length; i++) {
            wordToId.put(vocabulary[i], i + wordIdOffset);
        }

        this.vocabSize = vocabulary.length + wordIdOffset;
    }

    /**
     * Constructs a tokenizer from a pre-built vocabulary array.
     * Used by {@link #fromVocabFile(String)} to reconstruct a saved tokenizer.
     *
     * @param vocabulary words in vocabulary-ID order
     * @param includeSpecialTokens whether PAD/SOS/EOS are reserved at IDs 0–2
     */
    WordTokenizer(String[] vocabulary, boolean includeSpecialTokens) {
        this.includeSpecialTokens = includeSpecialTokens;
        this.wordIdOffset = includeSpecialTokens ? 3 : 0;
        this.vocabulary = vocabulary.clone();
        this.wordToId = new HashMap<>();
        for (int i = 0; i < vocabulary.length; i++) {
            wordToId.put(vocabulary[i], i + wordIdOffset);
        }
        this.vocabSize = vocabulary.length + wordIdOffset;
    }

    /**
     * Builds a tokenizer from the given text without special tokens.
     * Word IDs start at 0.
     */
    public static WordTokenizer fromText(String text) {
        return new WordTokenizer(text, false);
    }

    /**
     * Builds a tokenizer from the given text, optionally reserving IDs
     * for PAD (0), SOS (1), and EOS (2) special tokens.
     */
    public static WordTokenizer fromText(String text, boolean includeSpecialTokens) {
        return new WordTokenizer(text, includeSpecialTokens);
    }

    /**
     * Encodes a string into an array of integer token IDs by splitting on whitespace.
     *
     * @throws IllegalArgumentException if the string contains a word not in the vocabulary
     */
    @Override
    public int[] encode(String text) {
        String[] words = text.split("\\s+");

        // Filter out empty strings from leading/trailing whitespace
        List<String> filtered = new ArrayList<>();
        for (String word : words) {
            if (!word.isEmpty()) {
                filtered.add(word);
            }
        }

        int[] ids = new int[filtered.size()];
        for (int i = 0; i < filtered.size(); i++) {
            String word = filtered.get(i);
            Integer id = wordToId.get(word);
            if (id == null) {
                throw new IllegalArgumentException(
                        "Word '" + word + "' not found in vocabulary");
            }
            ids[i] = id;
        }
        return ids;
    }

    /**
     * Decodes an array of integer token IDs back into a string.
     * Words are joined with a single space. Special tokens (PAD, SOS, EOS)
     * are skipped during decoding.
     */
    @Override
    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for (int id : ids) {
            if (includeSpecialTokens && id < wordIdOffset) {
                // Skip special tokens in decoded output
                continue;
            }
            int wordIndex = id - wordIdOffset;
            if (wordIndex < 0 || wordIndex >= vocabulary.length) {
                throw new IllegalArgumentException(
                        "Token ID " + id + " is out of vocabulary range");
            }
            if (!first) {
                sb.append(' ');
            }
            sb.append(vocabulary[wordIndex]);
            first = false;
        }
        return sb.toString();
    }

    @Override
    public int getVocabSize() {
        return vocabSize;
    }

    @Override
    public boolean hasSpecialTokens() {
        return includeSpecialTokens;
    }

    /**
     * Returns the word ID offset (0 without special tokens, 3 with them).
     */
    public int getWordIdOffset() {
        return wordIdOffset;
    }

    /**
     * Saves the vocabulary to a plain-text file.
     *
     * The first line is a header declaring the tokenizer type and special-token
     * setting. Each subsequent line is one word from the vocabulary in ID order.
     */
    public void saveVocab(String path) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(Path.of(path))) {
            writer.write("#type=word special=" + includeSpecialTokens);
            writer.newLine();
            for (String word : vocabulary) {
                writer.write(word);
                writer.newLine();
            }
        }
    }

    /**
     * Reconstructs a WordTokenizer from a previously saved vocab file.
     *
     * @param path path to the vocab file
     * @return a WordTokenizer with the same vocabulary and settings
     * @throws IOException if the file cannot be read or has an invalid format
     */
    public static WordTokenizer fromVocabFile(String path) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(Path.of(path))) {
            String header = reader.readLine();
            if (header == null || !header.startsWith("#type=word")) {
                throw new IOException(
                        "Invalid vocab file header: expected '#type=word ...', got: " + header);
            }
            boolean special = header.contains("special=true");

            List<String> words = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                words.add(line);
            }

            return new WordTokenizer(words.toArray(new String[0]), special);
        }
    }
}
