package org.ea.javallm;

import org.ea.javallm.autograd.Tensor;
import org.ea.javallm.data.CharTokenizer;
import org.ea.javallm.data.ReversalTaskGenerator;
import org.ea.javallm.data.Tokenizer;
import org.ea.javallm.model.EncoderDecoderModel;
import org.ea.javallm.trainers.AdamOptimizer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Random;

/**
 * End-to-end encoder-decoder Transformer demo: trains on synthetic string
 * reversal, evaluates accuracy, and offers interactive mode.
 *
 * The reversal task is a simple sequence-to-sequence problem that demonstrates
 * the encoder-decoder architecture without requiring external data: given "abc",
 * produce "cba".
 *
 * Usage:
 *   javac -d out $(find src -name "*.java")
 *   java -cp out org.ea.javallm.TranslationTest
 */
public class TranslationTest {

    // --- Model hyperparameters (from SPECIFICATION.md Appendix C) ---
    private static final int EMBED_DIM = 64;
    private static final int NUM_LAYERS = 2;
    private static final int NUM_HEADS = 2;
    private static final int FFN_INNER_DIM = 256;
    private static final int MAX_SEQ_LEN = 32;

    // --- Training hyperparameters ---
    private static final double LEARNING_RATE = 1e-3;
    private static final int BATCH_SIZE = 16;
    private static final int TRAINING_STEPS = 3000;
    private static final int LOG_INTERVAL = 100;
    private static final int EVAL_INTERVAL = 1000;

    // --- Task hyperparameters ---
    // Using fixed-length strings avoids PAD token dilution in the cross-entropy
    // loss, which would otherwise make the loss misleadingly low while the model
    // hasn't actually learned the reversal task well.
    private static final int MIN_STRING_LEN = 5;
    private static final int MAX_STRING_LEN = 5;
    private static final int EVAL_EXAMPLES = 50;

    private static final String ALPHABET = "abcdefghijklmnopqrstuvwxyz";

    public static void main(String[] args) {
        try {
            run();
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void run() throws IOException {
        Random rng = new Random(42);

        System.out.println("=== String Reversal (Encoder-Decoder) Demo ===");
        System.out.println();

        // --- Build tokenizer and data generator ---
        Tokenizer tokenizer = CharTokenizer.fromText(ALPHABET, true);
        System.out.println("Vocabulary size: " + tokenizer.getVocabSize()
                + " (26 chars + 3 special tokens)");

        ReversalTaskGenerator generator = new ReversalTaskGenerator(
                tokenizer, MIN_STRING_LEN, MAX_STRING_LEN, rng);

        // --- Build model ---
        int vocabSize = tokenizer.getVocabSize();
        EncoderDecoderModel model = new EncoderDecoderModel(
                vocabSize, vocabSize, EMBED_DIM, NUM_LAYERS, NUM_HEADS,
                FFN_INNER_DIM, MAX_SEQ_LEN, rng);

        int paramCount = 0;
        for (Tensor p : model.getParameters()) {
            paramCount += p.size();
        }
        System.out.println("Model parameters: " + paramCount);

        AdamOptimizer optimizer = new AdamOptimizer(model.getParameters(), LEARNING_RATE);

        // --- Training loop ---
        System.out.println();
        System.out.println("Training for " + TRAINING_STEPS + " steps...");
        System.out.println();

        for (int step = 1; step <= TRAINING_STEPS; step++) {
            ReversalTaskGenerator.ReversalBatch batch = generator.generateBatch(BATCH_SIZE);

            int[][] source = batch.getSource();
            int[][] tgtInput = batch.getTgtInput();
            int[][] tgtTarget = batch.getTgtTarget();

            // Forward: logits shape (batch, tgtSeqLen, vocabSize)
            Tensor logits = model.forward(source, tgtInput);

            // Flatten for cross-entropy
            int batchSize = tgtInput.length;
            int tgtSeqLen = tgtInput[0].length;

            Tensor logitsFlat = logits.reshape(batchSize * tgtSeqLen, vocabSize);
            Tensor targetsFlat = targetBatchToTensor(tgtTarget);

            Tensor loss = logitsFlat.crossEntropy(targetsFlat);

            // Backward and update
            optimizer.zeroGrad();
            loss.backward();
            optimizer.step();

            if (step % LOG_INTERVAL == 0) {
                System.out.printf("Step %d/%d  loss=%.4f%n", step, TRAINING_STEPS,
                        loss.getData()[0]);
            }

            if (step % EVAL_INTERVAL == 0) {
                System.out.println();
                evaluateAndPrint(model, tokenizer, rng, 10);
                System.out.println();
            }
        }

        // --- Final evaluation ---
        System.out.println("=== Final Evaluation ===");
        System.out.println();
        double accuracy = evaluateAndPrint(model, tokenizer, rng, EVAL_EXAMPLES);
        System.out.println();
        System.out.printf("Final accuracy: %.1f%% on %d examples%n",
                accuracy * 100, EVAL_EXAMPLES);
        System.out.println();

        // --- Interactive mode ---
        interactiveMode(model, tokenizer);
    }

    /**
     * Greedy autoregressive decoding for the encoder-decoder model.
     *
     * Encodes the source once, then generates target tokens one at a time:
     * starts with SOS, feeds through the model, takes argmax of the last
     * position logits, appends the token, and repeats until EOS or maxLen.
     */
    private static String greedyDecode(EncoderDecoderModel model, Tokenizer tokenizer,
                                       String input) {
        int[] srcTokens = tokenizer.encode(input);
        int vocabSize = tokenizer.getVocabSize();

        int[][] srcBatch = new int[1][srcTokens.length];
        System.arraycopy(srcTokens, 0, srcBatch[0], 0, srcTokens.length);

        // Start decoder input with just SOS
        int maxTargetLen = srcTokens.length + 2; // room for reversed string + EOS
        int[] decoderTokens = new int[maxTargetLen];
        decoderTokens[0] = Tokenizer.SOS;
        int decoderLen = 1;

        StringBuilder result = new StringBuilder();

        for (int step = 0; step < maxTargetLen - 1; step++) {
            int[][] tgtBatch = new int[1][decoderLen];
            System.arraycopy(decoderTokens, 0, tgtBatch[0], 0, decoderLen);

            Tensor logits = model.forward(srcBatch, tgtBatch);

            // Get logits for the last decoder position
            double[] logitData = logits.getData();
            int lastPosOffset = (decoderLen - 1) * vocabSize;

            // Argmax
            int bestToken = 0;
            double bestScore = logitData[lastPosOffset];
            for (int i = 1; i < vocabSize; i++) {
                if (logitData[lastPosOffset + i] > bestScore) {
                    bestScore = logitData[lastPosOffset + i];
                    bestToken = i;
                }
            }

            if (bestToken == Tokenizer.EOS) break;

            decoderTokens[decoderLen] = bestToken;
            decoderLen++;

            // Decode non-special tokens (IDs 0-2 are PAD, SOS, EOS)
            if (bestToken >= 3) {
                result.append(tokenizer.decode(new int[]{bestToken}));
            }
        }

        return result.toString();
    }

    /**
     * Evaluates the model on randomly generated reversal examples and prints results.
     * Returns the fraction of examples where the model produced the exact correct reversal.
     */
    private static double evaluateAndPrint(EncoderDecoderModel model, Tokenizer tokenizer,
                                           Random rng, int numExamples) {
        int correct = 0;
        int printLimit = Math.min(numExamples, 5);

        for (int i = 0; i < numExamples; i++) {
            int len = MIN_STRING_LEN + rng.nextInt(MAX_STRING_LEN - MIN_STRING_LEN + 1);
            String input = generateRandomString(tokenizer, len, rng);
            String expected = new StringBuilder(input).reverse().toString();
            String predicted = greedyDecode(model, tokenizer, input);

            if (predicted.equals(expected)) {
                correct++;
            }

            if (i < printLimit) {
                String mark = predicted.equals(expected) ? "OK" : "WRONG";
                System.out.printf("  \"%s\" → \"%s\" (expected \"%s\") [%s]%n",
                        input, predicted, expected, mark);
            }
        }

        return (double) correct / numExamples;
    }

    /**
     * Interactive mode: user types a string, model reverses it, prints result vs correct.
     */
    private static void interactiveMode(EncoderDecoderModel model, Tokenizer tokenizer)
            throws IOException {
        System.out.println("=== Interactive Mode ===");
        System.out.println("Type a lowercase string to reverse. Type 'quit' to exit.");
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

            String predicted = greedyDecode(model, tokenizer, line);
            String expected = new StringBuilder(line).reverse().toString();
            System.out.println("  Model:    " + predicted);
            System.out.println("  Expected: " + expected);
            System.out.println("  " + (predicted.equals(expected) ? "Correct!" : "Incorrect"));
            System.out.println();
        }
    }

    /**
     * Converts a 2D target batch into a flat Tensor for cross-entropy.
     */
    private static Tensor targetBatchToTensor(int[][] targetBatch) {
        int batch = targetBatch.length;
        int seqLen = targetBatch[0].length;
        double[] data = new double[batch * seqLen];
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seqLen; s++) {
                data[b * seqLen + s] = targetBatch[b][s];
            }
        }
        return new Tensor(data, new int[]{batch * seqLen}, false);
    }

    private static String generateRandomString(Tokenizer tokenizer, int length, Random rng) {
        // Content token IDs start after the 3 special tokens (PAD, SOS, EOS)
        int offset = 3;
        int numChars = tokenizer.getVocabSize() - offset;
        StringBuilder sb = new StringBuilder(length);
        for (int i = 0; i < length; i++) {
            int charId = offset + rng.nextInt(numChars);
            sb.append(tokenizer.decode(new int[]{charId}));
        }
        return sb.toString();
    }
}
