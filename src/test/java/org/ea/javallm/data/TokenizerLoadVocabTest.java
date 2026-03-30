package org.ea.javallm.data;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class TokenizerLoadVocabTest {

    @TempDir
    Path tempDir;

    @Test
    void dispatchesToCharTokenizer() throws IOException {
        CharTokenizer original = CharTokenizer.fromText("hello");
        String vocabPath = tempDir.resolve("char.vocab").toString();
        original.saveVocab(vocabPath);

        Tokenizer loaded = Tokenizer.loadVocab(vocabPath);

        assertInstanceOf(CharTokenizer.class, loaded);
        assertEquals(original.getVocabSize(), loaded.getVocabSize());
        assertArrayEquals(original.encode("hello"), loaded.encode("hello"));
    }

    @Test
    void dispatchesToWordTokenizer() throws IOException {
        WordTokenizer original = WordTokenizer.fromText("hello world", true);
        String vocabPath = tempDir.resolve("word.vocab").toString();
        original.saveVocab(vocabPath);

        Tokenizer loaded = Tokenizer.loadVocab(vocabPath);

        assertInstanceOf(WordTokenizer.class, loaded);
        assertTrue(loaded.hasSpecialTokens());
        assertEquals(original.getVocabSize(), loaded.getVocabSize());
        assertArrayEquals(original.encode("hello world"), loaded.encode("hello world"));
    }

    @Test
    void vocabPathForModelReplacesExtension() {
        assertEquals("out.vocab", Tokenizer.vocabPathForModel("out.model"));
        assertEquals("path/to/my.vocab", Tokenizer.vocabPathForModel("path/to/my.model"));
    }

    @Test
    void vocabPathForModelAppendsWhenNoExtension() {
        assertEquals("mymodel.vocab", Tokenizer.vocabPathForModel("mymodel"));
        assertEquals("path/to/model.vocab", Tokenizer.vocabPathForModel("path/to/model"));
    }

    @Test
    void vocabPathForModelHandlesDotInDirectory() {
        // A dot in the directory name should not be mistaken for a file extension
        assertEquals("my.dir/model.vocab", Tokenizer.vocabPathForModel("my.dir/model"));
    }
}
