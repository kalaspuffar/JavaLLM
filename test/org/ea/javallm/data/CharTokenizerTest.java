package org.ea.javallm.data;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CharTokenizerTest {

    @Test
    void encodeDecodeRoundTrip() {
        CharTokenizer tok = CharTokenizer.fromText("hello");
        String original = "hello";
        int[] encoded = tok.encode(original);
        String decoded = tok.decode(encoded);
        assertEquals(original, decoded);
    }

    @Test
    void vocabularyIsSortedDeterministically() {
        CharTokenizer tok = CharTokenizer.fromText("cab");
        // Sorted order: a=0, b=1, c=2
        int[] encoded = tok.encode("abc");
        assertArrayEquals(new int[]{0, 1, 2}, encoded);
    }

    @Test
    void specialTokensShiftCharacterIds() {
        CharTokenizer tok = CharTokenizer.fromText("ab", true);
        // PAD=0, SOS=1, EOS=2, 'a'=3, 'b'=4
        assertEquals(5, tok.getVocabSize());
        int[] encoded = tok.encode("ab");
        assertArrayEquals(new int[]{3, 4}, encoded);
    }

    @Test
    void vocabSizeWithoutSpecialTokens() {
        CharTokenizer tok = CharTokenizer.fromText("hello");
        // Unique chars: e, h, l, o → 4
        assertEquals(4, tok.getVocabSize());
    }

    @Test
    void vocabSizeWithSpecialTokens() {
        CharTokenizer tok = CharTokenizer.fromText("hello", true);
        // 3 special + 4 unique chars = 7
        assertEquals(7, tok.getVocabSize());
    }

    @Test
    void decodeSkipsSpecialTokens() {
        CharTokenizer tok = CharTokenizer.fromText("ab", true);
        // Decode array containing SOS, 'a', 'b', EOS, PAD
        int[] ids = {CharTokenizer.SOS, 3, 4, CharTokenizer.EOS, CharTokenizer.PAD};
        assertEquals("ab", tok.decode(ids));
    }

    @Test
    void encodeUnknownCharacterThrows() {
        CharTokenizer tok = CharTokenizer.fromText("ab");
        assertThrows(IllegalArgumentException.class, () -> tok.encode("abc"));
    }

    @Test
    void encodeDecodeRoundTripWithSpecialTokens() {
        CharTokenizer tok = CharTokenizer.fromText("hello world", true);
        String original = "hello world";
        int[] encoded = tok.encode(original);
        String decoded = tok.decode(encoded);
        assertEquals(original, decoded);
    }
}
