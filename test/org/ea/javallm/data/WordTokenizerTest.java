package org.ea.javallm.data;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class WordTokenizerTest {

    @Test
    void vocabularyBuiltFromUniqueWords() {
        WordTokenizer tok = WordTokenizer.fromText("the cat sat on the mat");
        // Unique words: cat, mat, on, sat, the → 5
        assertEquals(5, tok.getVocabSize());
    }

    @Test
    void vocabularyIsSortedDeterministically() {
        WordTokenizer tok = WordTokenizer.fromText("banana apple cherry");
        // Sorted: apple=0, banana=1, cherry=2
        int[] encoded = tok.encode("apple banana cherry");
        assertArrayEquals(new int[]{0, 1, 2}, encoded);
    }

    @Test
    void encodeDecodeRoundTrip() {
        WordTokenizer tok = WordTokenizer.fromText("the cat sat on the mat");
        String original = "the cat sat";
        int[] encoded = tok.encode(original);
        String decoded = tok.decode(encoded);
        assertEquals(original, decoded);
    }

    @Test
    void encodeKnownSentence() {
        WordTokenizer tok = WordTokenizer.fromText("the cat sat");
        // Sorted: cat=0, sat=1, the=2
        int[] encoded = tok.encode("the cat sat");
        assertArrayEquals(new int[]{2, 0, 1}, encoded);
    }

    @Test
    void encodeUnknownWordThrows() {
        WordTokenizer tok = WordTokenizer.fromText("hello world");
        assertThrows(IllegalArgumentException.class, () -> tok.encode("hello universe"));
    }

    @Test
    void specialTokensShiftWordIds() {
        WordTokenizer tok = WordTokenizer.fromText("apple banana", true);
        // PAD=0, SOS=1, EOS=2, apple=3, banana=4
        assertEquals(5, tok.getVocabSize());
        int[] encoded = tok.encode("apple banana");
        assertArrayEquals(new int[]{3, 4}, encoded);
    }

    @Test
    void decodeSkipsSpecialTokens() {
        WordTokenizer tok = WordTokenizer.fromText("hello world", true);
        // PAD=0, SOS=1, EOS=2, hello=3, world=4
        int[] ids = {Tokenizer.PAD, 3, 4, Tokenizer.EOS};
        assertEquals("hello world", tok.decode(ids));
    }

    @Test
    void leadingTrailingWhitespaceFiltered() {
        WordTokenizer tok = WordTokenizer.fromText("  hello world  ");
        assertEquals(2, tok.getVocabSize());
        int[] encoded = tok.encode("hello world");
        String decoded = tok.decode(encoded);
        assertEquals("hello world", decoded);
    }

    @Test
    void hasSpecialTokensReflectsConstruction() {
        assertFalse(WordTokenizer.fromText("test").hasSpecialTokens());
        assertTrue(WordTokenizer.fromText("test", true).hasSpecialTokens());
    }

    @Test
    void vocabSizeWithoutSpecialTokens() {
        WordTokenizer tok = WordTokenizer.fromText("one two three");
        assertEquals(3, tok.getVocabSize());
    }

    @Test
    void vocabSizeWithSpecialTokens() {
        WordTokenizer tok = WordTokenizer.fromText("one two three", true);
        // 3 special + 3 words = 6
        assertEquals(6, tok.getVocabSize());
    }

    @Test
    void encodeDecodeRoundTripWithSpecialTokens() {
        WordTokenizer tok = WordTokenizer.fromText("hello beautiful world", true);
        String original = "hello beautiful world";
        int[] encoded = tok.encode(original);
        String decoded = tok.decode(encoded);
        assertEquals(original, decoded);
    }
}
