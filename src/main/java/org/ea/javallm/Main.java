package org.ea.javallm;

import org.ea.javallm.autograd.Tensor;
import org.ea.javallm.data.CharTokenizer;
import org.ea.javallm.data.SequenceBatcher;
import org.ea.javallm.data.TextReader;
import org.ea.javallm.data.Tokenizer;
import org.ea.javallm.data.WordTokenizer;
import org.ea.javallm.model.DecoderOnlyModel;
import org.ea.javallm.trainers.AdamOptimizer;
import org.ea.javallm.trainers.ModelIO;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Command-line entry point for JavaLLM.
 *
 * Supports two subcommands:
 *   train    — train a decoder-only Transformer and save the model
 *   generate — load a saved model and generate text
 *
 * Usage:
 *   java -jar javallm.jar train --data data/shakespeare.txt [options]
 *   java -jar javallm.jar generate --model out.model --data data/shakespeare.txt [options]
 *   java -jar javallm.jar --help
 */
public class Main {

    public static void main(String[] args) {
        try {
            run(args);
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    private static void run(String[] args) throws IOException {
        if (args.length == 0 || "--help".equals(args[0]) || "-h".equals(args[0])) {
            printUsage();
            return;
        }

        String subcommand = args[0];
        Map<String, String> options = parseOptions(args, 1);

        switch (subcommand) {
            case "train" -> runTrain(options);
            case "generate" -> runGenerate(options);
            default -> {
                System.err.println("Unknown subcommand: " + subcommand);
                System.err.println();
                printUsage();
                System.exit(1);
            }
        }
    }

    private static void runTrain(Map<String, String> options) throws IOException {
        String dataPath = requireOption(options, "data", "train");
        String modelPath = getOption(options, "model", "out.model");
        String tokenizerType = getOption(options, "tokenizer", "char");

        int embedDim = getIntOption(options, "embed-dim", 64);
        int numLayers = getIntOption(options, "layers", 2);
        int numHeads = getIntOption(options, "heads", 2);
        int ffnDim = getIntOption(options, "ffn-dim", 256);
        int contextLen = getIntOption(options, "context-len", 32);
        int batchSize = getIntOption(options, "batch-size", 8);
        int steps = getIntOption(options, "steps", 500);
        double learningRate = getDoubleOption(options, "learning-rate", 3e-4);
        int maxSeqLen = contextLen * 2;

        // --- Load data ---
        System.out.println("=== JavaLLM Training ===");
        System.out.println();

        if (!Files.exists(Paths.get(dataPath))) {
            System.err.println("Training data not found: " + dataPath);
            System.exit(1);
        }

        TextReader reader = new TextReader(dataPath);
        String trainText = reader.getTrainText();
        System.out.println("Training text: " + trainText.length() + " characters");

        // --- Build tokenizer ---
        Tokenizer tokenizer = buildTokenizer(tokenizerType, trainText);
        System.out.println("Tokenizer: " + tokenizerType);
        System.out.println("Vocabulary size: " + tokenizer.getVocabSize());

        int[] tokens = tokenizer.encode(trainText);
        System.out.println("Token count: " + tokens.length);

        Random rng = new Random(42);
        SequenceBatcher batcher = new SequenceBatcher(tokens, contextLen, batchSize, rng);

        // --- Build model ---
        DecoderOnlyModel model = new DecoderOnlyModel(
                tokenizer.getVocabSize(), embedDim, numLayers, numHeads,
                ffnDim, maxSeqLen, rng);

        int paramCount = 0;
        for (Tensor p : model.getParameters()) {
            paramCount += p.size();
        }
        System.out.println("Model parameters: " + paramCount);

        AdamOptimizer optimizer = new AdamOptimizer(model.getParameters(), learningRate);
        TextGenerator textGenerator = new TextGenerator(model, tokenizer, maxSeqLen, rng);

        // --- Training loop ---
        System.out.println();
        System.out.println("Training for " + steps + " steps...");
        System.out.println();

        int logInterval = Math.max(1, steps / 50);
        int sampleInterval = Math.max(1, steps / 5);

        for (int step = 1; step <= steps; step++) {
            if (!batcher.hasNext()) {
                batcher.reset();
            }

            int[][] inputBatch = batcher.nextInputBatch();
            int[][] targetBatch = batcher.nextTargetBatch();

            Tensor logits = model.forward(inputBatch);

            int currentBatchSize = inputBatch.length;
            int seqLen = inputBatch[0].length;
            int vocabSize = tokenizer.getVocabSize();

            Tensor logitsFlat = logits.reshape(currentBatchSize * seqLen, vocabSize);
            Tensor targetsFlat = CharGenerationTest.targetBatchToTensor(targetBatch);

            Tensor loss = logitsFlat.crossEntropy(targetsFlat);

            optimizer.zeroGrad();
            loss.backward();
            optimizer.step();

            if (step % logInterval == 0) {
                System.out.printf("Step %d/%d  loss=%.4f%n", step, steps,
                        loss.getData()[0]);
            }

            if (step % sampleInterval == 0) {
                System.out.println();
                System.out.println("--- Sample at step " + step + " ---");
                String samplePrompt = tokenizerType.equals("word") ? "The " : "The ";
                String sample = textGenerator.generate(samplePrompt, 50, 0.8);
                System.out.println(sample);
                System.out.println("--- End sample ---");
                System.out.println();
            }
        }

        // --- Save model ---
        System.out.println("Saving model to " + modelPath + "...");
        ModelIO.save(model.getNamedParameters(), modelPath);
        System.out.println("Model saved.");

        // --- Save vocab file alongside model ---
        String vocabPath = Tokenizer.vocabPathForModel(modelPath);
        System.out.println("Saving vocabulary to " + vocabPath + "...");
        tokenizer.saveVocab(vocabPath);
        System.out.println("Vocabulary saved.");
    }

    private static void runGenerate(Map<String, String> options) throws IOException {
        String modelPath = requireOption(options, "model", "generate");
        String dataPath = options.get("data");
        String tokenizerType = getOption(options, "tokenizer", "char");
        String prompt = options.get("prompt");
        double temperature = getDoubleOption(options, "temperature", 0.8);
        int length = getIntOption(options, "length", 100);

        int embedDim = getIntOption(options, "embed-dim", 64);
        int numLayers = getIntOption(options, "layers", 2);
        int numHeads = getIntOption(options, "heads", 2);
        int ffnDim = getIntOption(options, "ffn-dim", 256);
        int maxSeqLen = getIntOption(options, "context-len", 32) * 2;

        // --- Load tokenizer: prefer vocab file, fall back to --data ---
        String vocabPath = Tokenizer.vocabPathForModel(modelPath);
        Tokenizer tokenizer;

        if (Files.exists(Paths.get(vocabPath))) {
            tokenizer = Tokenizer.loadVocab(vocabPath);
            System.out.println("Loaded tokenizer from " + vocabPath);
        } else if (dataPath != null) {
            if (!Files.exists(Paths.get(dataPath))) {
                System.err.println("Data file not found: " + dataPath);
                System.exit(1);
            }
            TextReader reader = new TextReader(dataPath);
            tokenizer = buildTokenizer(tokenizerType, reader.getTrainText());
        } else {
            System.err.println("Error: no .vocab file found at " + vocabPath
                    + " and --data was not provided.");
            System.err.println("Either train with a newer version to generate a .vocab file,");
            System.err.println("or provide --data to rebuild the tokenizer from the training data.");
            System.exit(1);
            return; // unreachable, but satisfies the compiler
        }

        // --- Load model ---
        if (!Files.exists(Paths.get(modelPath))) {
            System.err.println("Model file not found: " + modelPath);
            System.exit(1);
        }

        Random rng = new Random(42);
        DecoderOnlyModel model = new DecoderOnlyModel(
                tokenizer.getVocabSize(), embedDim, numLayers, numHeads,
                ffnDim, maxSeqLen, rng);
        ModelIO.load(model.getNamedParameters(), modelPath);

        System.out.println("Model loaded from " + modelPath);
        System.out.println("Tokenizer: " + tokenizerType + " (vocab size: "
                + tokenizer.getVocabSize() + ")");
        System.out.println();

        TextGenerator textGenerator = new TextGenerator(model, tokenizer, maxSeqLen, rng);

        if (prompt != null) {
            String result = textGenerator.generate(prompt, length, temperature);
            System.out.println(result);
        } else {
            textGenerator.interactiveMode(length, temperature);
        }
    }

    private static Tokenizer buildTokenizer(String type, String text) {
        return switch (type) {
            case "char" -> CharTokenizer.fromText(text);
            case "word" -> WordTokenizer.fromText(text);
            default -> {
                System.err.println("Unknown tokenizer type: " + type
                        + " (expected 'char' or 'word')");
                System.exit(1);
                yield null;
            }
        };
    }

    // --- Argument parsing ---

    private static Map<String, String> parseOptions(String[] args, int startIndex) {
        Map<String, String> options = new HashMap<>();
        for (int i = startIndex; i < args.length; i++) {
            String arg = args[i];
            if (arg.startsWith("--") && i + 1 < args.length) {
                String key = arg.substring(2);
                String value = args[++i];
                options.put(key, value);
            } else if (arg.equals("--help") || arg.equals("-h")) {
                printUsage();
                System.exit(0);
            }
        }
        return options;
    }

    private static String requireOption(Map<String, String> options, String key,
                                        String subcommand) {
        String value = options.get(key);
        if (value == null) {
            System.err.println("Error: --" + key + " is required for '" + subcommand + "'");
            System.exit(1);
        }
        return value;
    }

    private static String getOption(Map<String, String> options, String key,
                                    String defaultValue) {
        return options.getOrDefault(key, defaultValue);
    }

    private static int getIntOption(Map<String, String> options, String key,
                                    int defaultValue) {
        String value = options.get(key);
        if (value == null) return defaultValue;
        return Integer.parseInt(value);
    }

    private static double getDoubleOption(Map<String, String> options, String key,
                                          double defaultValue) {
        String value = options.get(key);
        if (value == null) return defaultValue;
        return Double.parseDouble(value);
    }

    private static void printUsage() {
        System.out.println("JavaLLM — From-scratch Transformer in pure Java");
        System.out.println();
        System.out.println("Usage: java -jar javallm.jar <command> [options]");
        System.out.println();
        System.out.println("Commands:");
        System.out.println("  train      Train a decoder-only Transformer model");
        System.out.println("  generate   Generate text from a saved model");
        System.out.println();
        System.out.println("Train options:");
        System.out.println("  --data <path>          Training data file (REQUIRED)");
        System.out.println("  --model <path>         Output model path (default: out.model)");
        System.out.println("  --tokenizer <type>     Tokenizer: char or word (default: char)");
        System.out.println("  --embed-dim <n>        Embedding dimension (default: 64)");
        System.out.println("  --layers <n>           Number of Transformer layers (default: 2)");
        System.out.println("  --heads <n>            Number of attention heads (default: 2)");
        System.out.println("  --ffn-dim <n>          Feed-forward inner dimension (default: 256)");
        System.out.println("  --context-len <n>      Training context length (default: 32)");
        System.out.println("  --batch-size <n>       Training batch size (default: 8)");
        System.out.println("  --steps <n>            Number of training steps (default: 500)");
        System.out.println("  --learning-rate <r>    Adam learning rate (default: 3e-4)");
        System.out.println();
        System.out.println("Generate options:");
        System.out.println("  --model <path>         Saved model path (REQUIRED)");
        System.out.println("  --data <path>          Data file to rebuild tokenizer");
        System.out.println("                         (optional if .vocab file exists next to model)");
        System.out.println("  --tokenizer <type>     Tokenizer: char or word (default: char)");
        System.out.println("                         (ignored when loading from .vocab file)");
        System.out.println("  --prompt <text>        Text to continue (omit for interactive mode)");
        System.out.println("  --temperature <t>      Sampling temperature (default: 0.8)");
        System.out.println("  --length <n>           Max tokens to generate (default: 100)");
        System.out.println("  --embed-dim <n>        Must match training value (default: 64)");
        System.out.println("  --layers <n>           Must match training value (default: 2)");
        System.out.println("  --heads <n>            Must match training value (default: 2)");
        System.out.println("  --ffn-dim <n>          Must match training value (default: 256)");
        System.out.println("  --context-len <n>      Must match training value (default: 32)");
    }
}
