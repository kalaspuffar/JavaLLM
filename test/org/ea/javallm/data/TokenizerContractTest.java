package org.ea.javallm.data;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies that both CharTokenizer and WordTokenizer satisfy the Tokenizer
 * interface contract when used polymorphically.
 */
class TokenizerContractTest {

    @Test
    void charTokenizerSatisfiesTokenizerContract() {
        Tokenizer tok = CharTokenizer.fromText("hello world");
        assertTokenizerContract(tok, "hello");
    }

    @Test
    void wordTokenizerSatisfiesTokenizerContract() {
        Tokenizer tok = WordTokenizer.fromText("hello world");
        assertTokenizerContract(tok, "hello world");
    }

    @Test
    void charTokenizerWithSpecialTokensSatisfiesContract() {
        Tokenizer tok = CharTokenizer.fromText("abc", true);
        assertTrue(tok.hasSpecialTokens());
        assertTrue(tok.getVocabSize() > 3);
        assertTokenizerContract(tok, "abc");
    }

    @Test
    void wordTokenizerWithSpecialTokensSatisfiesContract() {
        Tokenizer tok = WordTokenizer.fromText("hello world", true);
        assertTrue(tok.hasSpecialTokens());
        assertTrue(tok.getVocabSize() > 3);
        assertTokenizerContract(tok, "hello world");
    }

    @Test
    void bothTokenizersShareSpecialTokenConstants() {
        // Constants are defined on the interface and inherited by both implementations
        assertEquals(Tokenizer.PAD, 0);
        assertEquals(Tokenizer.SOS, 1);
        assertEquals(Tokenizer.EOS, 2);
    }

    /**
     * Asserts the core Tokenizer contract: encode produces valid IDs,
     * decode round-trips, and vocabSize is positive.
     */
    private void assertTokenizerContract(Tokenizer tok, String text) {
        assertTrue(tok.getVocabSize() > 0, "vocabSize must be positive");

        int[] ids = tok.encode(text);
        assertNotNull(ids);
        assertTrue(ids.length > 0, "encode must produce at least one token");

        for (int id : ids) {
            assertTrue(id >= 0 && id < tok.getVocabSize(),
                    "token ID " + id + " out of range [0, " + tok.getVocabSize() + ")");
        }

        String decoded = tok.decode(ids);
        assertEquals(text, decoded, "decode(encode(text)) must round-trip");
    }
}
