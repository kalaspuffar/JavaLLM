package org.ea.javallm;

import org.ea.javallm.autograd.Tensor;
import org.ea.javallm.data.CharTokenizer;
import org.ea.javallm.data.SequenceBatcher;
import org.ea.javallm.data.TextReader;
import org.ea.javallm.data.Tokenizer;
import org.ea.javallm.model.DecoderOnlyModel;
import org.ea.javallm.trainers.AdamOptimizer;
import org.ea.javallm.trainers.ModelIO;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;

/**
 * End-to-end decoder-only Transformer demo: trains a character-level language
 * model on Shakespeare text, generates samples during training to show
 * improvement, saves the trained model, and offers interactive generation.
 *
 * This demo uses hardcoded defaults. For configurable training, use the CLI:
 *   java -jar javallm.jar train --data data/shakespeare.txt
 *
 * Requires data/shakespeare.txt — run data/download-shakespeare.sh to obtain it.
 */
public class CharGenerationTest {

    // --- Model hyperparameters ---
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

            TextGenerator generator = new TextGenerator(model, tokenizer, MAX_SEQ_LEN, rng);
            generator.interactiveMode(GENERATE_LEN, TEMPERATURE);
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
        TextGenerator textGenerator = new TextGenerator(model, tokenizer, MAX_SEQ_LEN, rng);

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

            Tensor logits = model.forward(inputBatch);

            int batchSize = inputBatch.length;
            int seqLen = inputBatch[0].length;
            int vocabSize = tokenizer.getVocabSize();

            Tensor logitsFlat = logits.reshape(batchSize * seqLen, vocabSize);
            Tensor targetsFlat = targetBatchToTensor(targetBatch);

            Tensor loss = logitsFlat.crossEntropy(targetsFlat);

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
                String sample = textGenerator.generate("The ", GENERATE_LEN, TEMPERATURE);
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
        textGenerator.interactiveMode(GENERATE_LEN, TEMPERATURE);
    }

    /**
     * Converts a 2D target batch into a flat Tensor for cross-entropy.
     */
    static Tensor targetBatchToTensor(int[][] targetBatch) {
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
