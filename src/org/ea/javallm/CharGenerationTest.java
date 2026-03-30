package org.ea.javallm;

import org.ea.javallm.autograd.Tensor;
import org.ea.javallm.data.CharTokenizer;
import org.ea.javallm.data.SequenceBatcher;
import org.ea.javallm.data.TextReader;
import org.ea.javallm.data.Tokenizer;
import org.ea.javallm.model.DecoderOnlyModel;
import org.ea.javallm.trainers.AdamOptimizer;
import org.ea.javallm.trainers.ModelIO;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;

/**
 * End-to-end decoder-only Transformer demo: trains a character-level language
 * model on Shakespeare text, generates samples during training to show
 * improvement, saves the trained model, and offers interactive generation.
 *
 * Usage:
 *   javac -d out $(find src -name "*.java")
 *   java -cp out org.ea.javallm.CharGenerationTest
 *
 * Requires data/shakespeare.txt — run data/download-shakespeare.sh to obtain it.
 */
public class CharGenerationTest {

    // --- Model hyperparameters ---
    // Scaled down from spec Appendix C defaults (128/4/4/512) to be practical
    // for CPU training with the autograd engine's per-tensor memory overhead.
    // Increase these values if running with a larger heap (-Xmx).
    private static final int EMBED_DIM = 64;
    private static final int NUM_LAYERS = 2;
    private static final int NUM_HEADS = 2;
    private static final int FFN_INNER_DIM = 256;
    private static final int MAX_SEQ_LEN = 64;

    // --- Training hyperparameters ---
    private static final double LEARNING_RATE = 3e-4;
    private static final int BATCH_SIZE = 8;
    private static final int CONTEXT_LEN = 32;
    private static final int TRAINING_STEPS = 500;
    private static final int LOG_INTERVAL = 10;
    private static final int SAMPLE_INTERVAL = 100;

    // --- Generation hyperparameters ---
    private static final double TEMPERATURE = 0.8;
    private static final int GENERATE_LEN = 100;

    // --- File paths ---
    private static final String DATA_PATH = "data/shakespeare.txt";
    private static final String MODEL_PATH = "data/shakespeare.model";

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

        // Check for saved model
        if (Files.exists(Paths.get(MODEL_PATH))) {
            System.out.println("Found saved model at " + MODEL_PATH);
            System.out.println("Loading model and entering interactive mode...");
            System.out.println();

            TextReader reader = loadTextReader();
            Tokenizer tokenizer = CharTokenizer.fromText(reader.getTrainText());
            DecoderOnlyModel model = new DecoderOnlyModel(
                    tokenizer.getVocabSize(), EMBED_DIM, NUM_LAYERS, NUM_HEADS,
                    FFN_INNER_DIM, MAX_SEQ_LEN, rng);
            ModelIO.load(model.getNamedParameters(), MODEL_PATH);

            interactiveMode(model, tokenizer, rng);
            return;
        }

        // --- Load data ---
        System.out.println("=== Character-Level Language Model Demo ===");
        System.out.println();

        TextReader reader = loadTextReader();
        String trainText = reader.getTrainText();
        System.out.println("Training text: " + trainText.length() + " characters");

        Tokenizer tokenizer = CharTokenizer.fromText(trainText);
        System.out.println("Vocabulary size: " + tokenizer.getVocabSize());

        int[] tokens = tokenizer.encode(trainText);
        SequenceBatcher batcher = new SequenceBatcher(tokens, CONTEXT_LEN, BATCH_SIZE, rng);

        // --- Build model ---
        DecoderOnlyModel model = new DecoderOnlyModel(
                tokenizer.getVocabSize(), EMBED_DIM, NUM_LAYERS, NUM_HEADS,
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
            if (!batcher.hasNext()) {
                batcher.reset();
            }

            int[][] inputBatch = batcher.nextInputBatch();
            int[][] targetBatch = batcher.nextTargetBatch();

            // Forward: get logits (batch, seqLen, vocabSize)
            Tensor logits = model.forward(inputBatch);

            // Flatten for cross-entropy: (batch*seqLen, vocabSize) and (batch*seqLen)
            int batchSize = inputBatch.length;
            int seqLen = inputBatch[0].length;
            int vocabSize = tokenizer.getVocabSize();

            Tensor logitsFlat = logits.reshape(batchSize * seqLen, vocabSize);
            Tensor targetsFlat = targetBatchToTensor(targetBatch);

            Tensor loss = logitsFlat.crossEntropy(targetsFlat);

            // Backward and update
            optimizer.zeroGrad();
            loss.backward();
            optimizer.step();

            if (step % LOG_INTERVAL == 0) {
                System.out.printf("Step %d/%d  loss=%.4f%n", step, TRAINING_STEPS,
                        loss.getData()[0]);
            }

            if (step % SAMPLE_INTERVAL == 0) {
                System.out.println();
                System.out.println("--- Sample at step " + step + " ---");
                String sample = generateText(model, tokenizer, "The ", GENERATE_LEN,
                        TEMPERATURE, rng);
                System.out.println(sample);
                System.out.println("--- End sample ---");
                System.out.println();
            }
        }

        // --- Save model ---
        System.out.println("Saving model to " + MODEL_PATH + "...");
        ModelIO.save(model.getNamedParameters(), MODEL_PATH);
        System.out.println("Model saved.");
        System.out.println();

        // --- Interactive mode ---
        interactiveMode(model, tokenizer, rng);
    }

    /**
     * Autoregressive text generation with temperature sampling.
     *
     * Uses a sliding window of the last maxSeqLen tokens as context. At each step,
     * computes logits for the last position, scales by temperature, applies softmax,
     * and samples from the distribution.
     */
    private static String generateText(DecoderOnlyModel model, Tokenizer tokenizer,
                                       String prompt, int maxLen, double temperature,
                                       Random rng) {
        int[] promptTokens = tokenizer.encode(prompt);
        int vocabSize = tokenizer.getVocabSize();

        // Build the full token sequence, starting from the prompt
        int[] generated = new int[promptTokens.length + maxLen];
        System.arraycopy(promptTokens, 0, generated, 0, promptTokens.length);
        int totalLen = promptTokens.length;

        for (int i = 0; i < maxLen; i++) {
            // Sliding window: use the last MAX_SEQ_LEN tokens as context
            int contextStart = Math.max(0, totalLen - MAX_SEQ_LEN);
            int contextLen = totalLen - contextStart;

            int[][] input = new int[1][contextLen];
            System.arraycopy(generated, contextStart, input[0], 0, contextLen);

            Tensor logits = model.forward(input);

            // Extract logits for the last position: shape (1, contextLen, vocabSize)
            double[] logitData = logits.getData();
            int lastPosOffset = (contextLen - 1) * vocabSize;

            // Temperature scaling and softmax sampling
            int sampledToken = sampleFromLogits(logitData, lastPosOffset, vocabSize,
                    temperature, rng);

            generated[totalLen] = sampledToken;
            totalLen++;
        }

        return tokenizer.decode(java.util.Arrays.copyOf(generated, totalLen));
    }

    /**
     * Samples a token from logits at a given offset using temperature-scaled softmax.
     */
    private static int sampleFromLogits(double[] logits, int offset, int vocabSize,
                                        double temperature, Random rng) {
        // Temperature scaling
        double[] scaled = new double[vocabSize];
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < vocabSize; i++) {
            scaled[i] = logits[offset + i] / temperature;
            if (scaled[i] > max) max = scaled[i];
        }

        // Softmax
        double sumExp = 0.0;
        for (int i = 0; i < vocabSize; i++) {
            scaled[i] = Math.exp(scaled[i] - max);
            sumExp += scaled[i];
        }
        for (int i = 0; i < vocabSize; i++) {
            scaled[i] /= sumExp;
        }

        // Multinomial sampling
        double r = rng.nextDouble();
        double cumulative = 0.0;
        for (int i = 0; i < vocabSize; i++) {
            cumulative += scaled[i];
            if (r < cumulative) return i;
        }
        return vocabSize - 1;
    }

    /**
     * Interactive generation loop: reads prompts from stdin and generates continuations.
     */
    private static void interactiveMode(DecoderOnlyModel model, Tokenizer tokenizer,
                                        Random rng) throws IOException {
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

            String result = generateText(model, tokenizer, line, GENERATE_LEN,
                    TEMPERATURE, rng);
            System.out.println(result);
            System.out.println();
        }
    }

    /**
     * Converts a 2D target batch into a flat Tensor for cross-entropy.
     * Target shape (batch, seqLen) → flat Tensor shape (batch*seqLen).
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

    private static TextReader loadTextReader() throws IOException {
        if (!Files.exists(Paths.get(DATA_PATH))) {
            System.err.println("Training data not found at " + DATA_PATH);
            System.err.println("Run: data/download-shakespeare.sh");
            System.exit(1);
        }
        return new TextReader(DATA_PATH);
    }
}
