package org.ea.javallm.data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeSet;

/**
 * Character-level tokenizer that builds a vocabulary from input text.
 *
 * Characters are sorted to ensure deterministic ID assignment regardless of
 * their order in the training text. When special tokens are enabled (for
 * encoder-decoder tasks), PAD=0, SOS=1, EOS=2 are reserved and character
 * IDs start at 3.
 */
public class CharTokenizer implements Tokenizer {

    private final char[] vocabulary;
    private final Map<Character, Integer> charToId;
    private final int vocabSize;
    private final boolean includeSpecialTokens;
    private final int characterIdOffset;

    private CharTokenizer(String text, boolean includeSpecialTokens) {
        this.includeSpecialTokens = includeSpecialTokens;
        this.characterIdOffset = includeSpecialTokens ? 3 : 0;

        // Collect unique characters in sorted order for deterministic ID assignment
        TreeSet<Character> uniqueChars = new TreeSet<>();
        for (char c : text.toCharArray()) {
            uniqueChars.add(c);
        }

        this.vocabulary = new char[uniqueChars.size()];
        this.charToId = new HashMap<>();
        int index = 0;
        for (char c : uniqueChars) {
            vocabulary[index] = c;
            charToId.put(c, index + characterIdOffset);
            index++;
        }

        this.vocabSize = vocabulary.length + characterIdOffset;
    }

    /**
     * Constructs a tokenizer from a pre-built vocabulary array.
     * Used by {@link #fromVocabFile(String)} to reconstruct a saved tokenizer.
     *
     * @param vocabulary characters in vocabulary-ID order
     * @param includeSpecialTokens whether PAD/SOS/EOS are reserved at IDs 0–2
     */
    CharTokenizer(char[] vocabulary, boolean includeSpecialTokens) {
        this.includeSpecialTokens = includeSpecialTokens;
        this.characterIdOffset = includeSpecialTokens ? 3 : 0;
        this.vocabulary = Arrays.copyOf(vocabulary, vocabulary.length);
        this.charToId = new HashMap<>();
        for (int i = 0; i < vocabulary.length; i++) {
            charToId.put(vocabulary[i], i + characterIdOffset);
        }
        this.vocabSize = vocabulary.length + characterIdOffset;
    }

    /**
     * Builds a tokenizer from the given text without special tokens.
     * Character IDs start at 0.
     */
    public static CharTokenizer fromText(String text) {
        return new CharTokenizer(text, false);
    }

    /**
     * Builds a tokenizer from the given text, optionally reserving IDs
     * for PAD (0), SOS (1), and EOS (2) special tokens.
     */
    public static CharTokenizer fromText(String text, boolean includeSpecialTokens) {
        return new CharTokenizer(text, includeSpecialTokens);
    }

    /**
     * Encodes a string into an array of integer token IDs.
     *
     * @throws IllegalArgumentException if the string contains a character not in the vocabulary
     */
    public int[] encode(String text) {
        int[] ids = new int[text.length()];
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            Integer id = charToId.get(c);
            if (id == null) {
                throw new IllegalArgumentException(
                        "Character '" + c + "' not found in vocabulary");
            }
            ids[i] = id;
        }
        return ids;
    }

    /**
     * Decodes an array of integer token IDs back into a string.
     * Special tokens (PAD, SOS, EOS) are skipped during decoding.
     */
    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder(ids.length);
        for (int id : ids) {
            if (includeSpecialTokens && id < characterIdOffset) {
                // Skip special tokens in decoded output
                continue;
            }
            int charIndex = id - characterIdOffset;
            if (charIndex < 0 || charIndex >= vocabulary.length) {
                throw new IllegalArgumentException(
                        "Token ID " + id + " is out of vocabulary range");
            }
            sb.append(vocabulary[charIndex]);
        }
        return sb.toString();
    }

    /**
     * Returns the total vocabulary size, including special tokens if enabled.
     */
    public int getVocabSize() {
        return vocabSize;
    }

    /**
     * Returns whether this tokenizer includes special tokens (PAD, SOS, EOS).
     */
    public boolean hasSpecialTokens() {
        return includeSpecialTokens;
    }

    /**
     * Returns the character ID offset (0 without special tokens, 3 with them).
     */
    public int getCharacterIdOffset() {
        return characterIdOffset;
    }

    /**
     * Saves the vocabulary to a plain-text file.
     *
     * The first line is a header declaring the tokenizer type and special-token
     * setting. Each subsequent line is one character from the vocabulary in ID
     * order. Whitespace characters are escaped: space → {@code \s},
     * newline → {@code \n}, tab → {@code \t}, backslash → {@code \\}.
     */
    public void saveVocab(String path) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(Path.of(path))) {
            writer.write("#type=char special=" + includeSpecialTokens);
            writer.newLine();
            for (char c : vocabulary) {
                writer.write(escapeChar(c));
                writer.newLine();
            }
        }
    }

    /**
     * Reconstructs a CharTokenizer from a previously saved vocab file.
     *
     * @param path path to the vocab file
     * @return a CharTokenizer with the same vocabulary and settings
     * @throws IOException if the file cannot be read or has an invalid format
     */
    public static CharTokenizer fromVocabFile(String path) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(Path.of(path))) {
            String header = reader.readLine();
            if (header == null || !header.startsWith("#type=char")) {
                throw new IOException(
                        "Invalid vocab file header: expected '#type=char ...', got: " + header);
            }
            boolean special = header.contains("special=true");

            java.util.List<Character> chars = new java.util.ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                chars.add(unescapeChar(line));
            }

            char[] vocab = new char[chars.size()];
            for (int i = 0; i < chars.size(); i++) {
                vocab[i] = chars.get(i);
            }
            return new CharTokenizer(vocab, special);
        }
    }

    private static String escapeChar(char c) {
        return switch (c) {
            case ' ' -> "\\s";
            case '\n' -> "\\n";
            case '\t' -> "\\t";
            case '\\' -> "\\\\";
            default -> String.valueOf(c);
        };
    }

    private static char unescapeChar(String token) {
        return switch (token) {
            case "\\s" -> ' ';
            case "\\n" -> '\n';
            case "\\t" -> '\t';
            case "\\\\" -> '\\';
            default -> {
                if (token.length() != 1) {
                    throw new IllegalArgumentException(
                            "Expected single character or escape sequence, got: '" + token + "'");
                }
                yield token.charAt(0);
            }
        };
    }
}
