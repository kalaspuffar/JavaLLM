package org.ea.javallm.data;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class WordTokenizerVocabTest {

    @TempDir
    Path tempDir;

    @Test
    void saveAndLoadRoundTrip() throws IOException {
        WordTokenizer original = WordTokenizer.fromText("the cat sat on the mat");
        String vocabPath = tempDir.resolve("test.vocab").toString();

        original.saveVocab(vocabPath);
        WordTokenizer loaded = WordTokenizer.fromVocabFile(vocabPath);

        assertEquals(original.getVocabSize(), loaded.getVocabSize());
        assertEquals(original.hasSpecialTokens(), loaded.hasSpecialTokens());

        String text = "the cat sat";
        assertArrayEquals(original.encode(text), loaded.encode(text));
        assertEquals(text, loaded.decode(loaded.encode(text)));
    }

    @Test
    void saveAndLoadWithSpecialTokens() throws IOException {
        WordTokenizer original = WordTokenizer.fromText("apple banana cherry", true);
        String vocabPath = tempDir.resolve("special.vocab").toString();

        original.saveVocab(vocabPath);
        WordTokenizer loaded = WordTokenizer.fromVocabFile(vocabPath);

        assertTrue(loaded.hasSpecialTokens());
        assertEquals(original.getVocabSize(), loaded.getVocabSize());
        assertArrayEquals(original.encode("apple banana"), loaded.encode("apple banana"));
    }

    @Test
    void headerContainsTypeAndSpecialFlag() throws IOException {
        WordTokenizer tok = WordTokenizer.fromText("hello world", false);
        String vocabPath = tempDir.resolve("header.vocab").toString();
        tok.saveVocab(vocabPath);

        String firstLine = Files.readAllLines(Path.of(vocabPath)).get(0);
        assertEquals("#type=word special=false", firstLine);
    }

    @Test
    void headerWithSpecialTokensTrue() throws IOException {
        WordTokenizer tok = WordTokenizer.fromText("hello world", true);
        String vocabPath = tempDir.resolve("header_special.vocab").toString();
        tok.saveVocab(vocabPath);

        String firstLine = Files.readAllLines(Path.of(vocabPath)).get(0);
        assertEquals("#type=word special=true", firstLine);
    }

    @Test
    void loadedTokenizerEncodesIdentically() throws IOException {
        String corpus = "the quick brown fox jumps over the lazy dog";
        WordTokenizer original = WordTokenizer.fromText(corpus);
        String vocabPath = tempDir.resolve("fox.vocab").toString();

        original.saveVocab(vocabPath);
        WordTokenizer loaded = WordTokenizer.fromVocabFile(vocabPath);

        assertArrayEquals(original.encode(corpus), loaded.encode(corpus));
        assertEquals(corpus, loaded.decode(loaded.encode(corpus)));
    }
}
