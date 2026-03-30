package org.ea.javallm;

import org.ea.javallm.autograd.Tensor;
import org.ea.javallm.data.Tokenizer;
import org.ea.javallm.model.DecoderOnlyModel;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Random;

/**
 * Shared text generation utilities for decoder-only models.
 *
 * Provides autoregressive generation with temperature sampling and an
 * interactive prompt-response loop. Used by both {@link CharGenerationTest}
 * and {@link Main} to avoid duplicating generation logic.
 */
public class TextGenerator {

    private final DecoderOnlyModel model;
    private final Tokenizer tokenizer;
    private final int maxSeqLen;
    private final Random rng;

    public TextGenerator(DecoderOnlyModel model, Tokenizer tokenizer, int maxSeqLen, Random rng) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.maxSeqLen = maxSeqLen;
        this.rng = rng;
    }

    /**
     * Generates text by continuing from the given prompt using temperature sampling.
     *
     * Uses a sliding window of the last {@code maxSeqLen} tokens as context. At each
     * step, computes logits for the last position, scales by temperature, applies
     * softmax, and samples from the distribution.
     */
    public String generate(String prompt, int maxLen, double temperature) {
        int[] promptTokens = tokenizer.encode(prompt);
        int vocabSize = tokenizer.getVocabSize();

        int[] generated = new int[promptTokens.length + maxLen];
        System.arraycopy(promptTokens, 0, generated, 0, promptTokens.length);
        int totalLen = promptTokens.length;

        for (int i = 0; i < maxLen; i++) {
            int contextStart = Math.max(0, totalLen - maxSeqLen);
            int contextLen = totalLen - contextStart;

            int[][] input = new int[1][contextLen];
            System.arraycopy(generated, contextStart, input[0], 0, contextLen);

            Tensor logits = model.forward(input);

            double[] logitData = logits.getData();
            int lastPosOffset = (contextLen - 1) * vocabSize;

            int sampledToken = sampleFromLogits(logitData, lastPosOffset, vocabSize,
                    temperature);

            generated[totalLen] = sampledToken;
            totalLen++;
        }

        return tokenizer.decode(Arrays.copyOf(generated, totalLen));
    }

    /**
     * Interactive generation loop: reads prompts from stdin and generates continuations.
     */
    public void interactiveMode(int generateLen, double temperature) throws IOException {
        System.out.println("=== Interactive Mode ===");
        System.out.println("Type a prompt and press Enter to generate text.");
        System.out.println("Type 'quit' to exit.");
        System.out.println();

        BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in));
        String line;
        while (true) {
            System.out.print("> ");
            System.out.flush();
            line = stdin.readLine();
            if (line == null || line.equalsIgnoreCase("quit")) {
                System.out.println("Goodbye.");
                break;
            }
            if (line.isEmpty()) continue;

            String result = generate(line, generateLen, temperature);
            System.out.println(result);
            System.out.println();
        }
    }

    /**
     * Samples a token from logits at a given offset using temperature-scaled softmax.
     */
    private int sampleFromLogits(double[] logits, int offset, int vocabSize,
                                 double temperature) {
        double[] scaled = new double[vocabSize];
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < vocabSize; i++) {
            scaled[i] = logits[offset + i] / temperature;
            if (scaled[i] > max) max = scaled[i];
        }

        double sumExp = 0.0;
        for (int i = 0; i < vocabSize; i++) {
            scaled[i] = Math.exp(scaled[i] - max);
            sumExp += scaled[i];
        }
        for (int i = 0; i < vocabSize; i++) {
            scaled[i] /= sumExp;
        }

        double r = rng.nextDouble();
        double cumulative = 0.0;
        for (int i = 0; i < vocabSize; i++) {
            cumulative += scaled[i];
            if (r < cumulative) return i;
        }
        return vocabSize - 1;
    }
}
