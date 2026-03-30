package org.ea.javallm.data;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Reads an entire text file and splits it into training and validation portions.
 *
 * The split is character-based: the first {@code trainSplit} fraction of the text
 * becomes the training set, and the remainder becomes the validation set. The
 * entire file is held in memory, which is acceptable for small corpora like
 * Tiny Shakespeare (~1MB).
 */
public class TextReader {

    private static final double DEFAULT_TRAIN_SPLIT = 0.9;

    private final String trainText;
    private final String validationText;

    /**
     * Reads the file and splits it at the default ratio (90% train, 10% validation).
     *
     * @param filePath path to the text file
     * @throws IOException if the file cannot be read
     */
    public TextReader(String filePath) throws IOException {
        this(filePath, DEFAULT_TRAIN_SPLIT);
    }

    /**
     * Reads the file and splits it at the given ratio.
     *
     * @param filePath   path to the text file
     * @param trainSplit fraction of the text to use for training (0.0 to 1.0)
     * @throws IOException              if the file cannot be read
     * @throws IllegalArgumentException if trainSplit is not between 0 and 1
     */
    public TextReader(String filePath, double trainSplit) throws IOException {
        if (trainSplit < 0.0 || trainSplit > 1.0) {
            throw new IllegalArgumentException(
                    "trainSplit must be between 0.0 and 1.0, got " + trainSplit);
        }
        String fullText = new String(Files.readAllBytes(Paths.get(filePath)));
        int splitIndex = (int) (fullText.length() * trainSplit);
        this.trainText = fullText.substring(0, splitIndex);
        this.validationText = fullText.substring(splitIndex);
    }

    public String getTrainText() {
        return trainText;
    }

    public String getValidationText() {
        return validationText;
    }
}
