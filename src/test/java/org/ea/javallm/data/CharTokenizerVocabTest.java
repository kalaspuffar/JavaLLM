package org.ea.javallm.data;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class CharTokenizerVocabTest {

    @TempDir
    Path tempDir;

    @Test
    void saveAndLoadRoundTrip() throws IOException {
        CharTokenizer original = CharTokenizer.fromText("hello world");
        String vocabPath = tempDir.resolve("test.vocab").toString();

        original.saveVocab(vocabPath);
        CharTokenizer loaded = CharTokenizer.fromVocabFile(vocabPath);

        assertEquals(original.getVocabSize(), loaded.getVocabSize());
        assertEquals(original.hasSpecialTokens(), loaded.hasSpecialTokens());

        String text = "hello world";
        assertArrayEquals(original.encode(text), loaded.encode(text));
        assertEquals(text, loaded.decode(loaded.encode(text)));
    }

    @Test
    void saveAndLoadWithSpecialTokens() throws IOException {
        CharTokenizer original = CharTokenizer.fromText("abc", true);
        String vocabPath = tempDir.resolve("special.vocab").toString();

        original.saveVocab(vocabPath);
        CharTokenizer loaded = CharTokenizer.fromVocabFile(vocabPath);

        assertTrue(loaded.hasSpecialTokens());
        assertEquals(original.getVocabSize(), loaded.getVocabSize());
        assertArrayEquals(original.encode("abc"), loaded.encode("abc"));
    }

    @Test
    void whitespaceEscaping() throws IOException {
        // Build a vocabulary containing space, newline, tab, and backslash
        CharTokenizer original = CharTokenizer.fromText("a \n\t\\");
        String vocabPath = tempDir.resolve("ws.vocab").toString();

        original.saveVocab(vocabPath);

        // Verify the file contents have the expected escape sequences
        String fileContent = Files.readString(Path.of(vocabPath));
        assertTrue(fileContent.contains("\\s"), "Space should be escaped as \\s");
        assertTrue(fileContent.contains("\\n"), "Newline should be escaped as \\n");
        assertTrue(fileContent.contains("\\t"), "Tab should be escaped as \\t");
        assertTrue(fileContent.contains("\\\\"), "Backslash should be escaped as \\\\");

        // Verify round-trip preserves the characters
        CharTokenizer loaded = CharTokenizer.fromVocabFile(vocabPath);
        assertEquals(original.getVocabSize(), loaded.getVocabSize());

        String text = "a \n\t\\";
        assertArrayEquals(original.encode(text), loaded.encode(text));
        assertEquals(text, loaded.decode(loaded.encode(text)));
    }

    @Test
    void headerContainsTypeAndSpecialFlag() throws IOException {
        CharTokenizer tok = CharTokenizer.fromText("ab", false);
        String vocabPath = tempDir.resolve("header.vocab").toString();
        tok.saveVocab(vocabPath);

        String firstLine = Files.readAllLines(Path.of(vocabPath)).get(0);
        assertEquals("#type=char special=false", firstLine);
    }

    @Test
    void headerWithSpecialTokensTrue() throws IOException {
        CharTokenizer tok = CharTokenizer.fromText("ab", true);
        String vocabPath = tempDir.resolve("header_special.vocab").toString();
        tok.saveVocab(vocabPath);

        String firstLine = Files.readAllLines(Path.of(vocabPath)).get(0);
        assertEquals("#type=char special=true", firstLine);
    }

    @Test
    void loadedTokenizerEncodesAndDecodesIdentically() throws IOException {
        String corpus = "The quick brown fox jumps over the lazy dog";
        CharTokenizer original = CharTokenizer.fromText(corpus);
        String vocabPath = tempDir.resolve("fox.vocab").toString();

        original.saveVocab(vocabPath);
        CharTokenizer loaded = CharTokenizer.fromVocabFile(vocabPath);

        // Every substring of the corpus should encode/decode identically
        for (String word : corpus.split(" ")) {
            assertArrayEquals(original.encode(word), loaded.encode(word),
                    "Encoding mismatch for: " + word);
        }
    }
}
