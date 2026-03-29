package org.ea.javallm.trainers;

import org.ea.javallm.autograd.Tensor;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Map;

/**
 * Saves and loads named model parameters in a human-readable plain text format.
 *
 * Format per tensor:
 * <pre>
 * --- name [dim0 dim1 ...]
 * value0 value1 value2 ...
 * </pre>
 *
 * Lines starting with '#' are treated as comments and ignored on load.
 * Blank lines between tensors are allowed.
 */
public class ModelIO {

    /**
     * Saves all named parameters to a plain text file.
     *
     * @param namedParameters map of parameter names to tensors (insertion order preserved)
     * @param filePath        path to the output file
     * @throws IOException if the file cannot be written
     */
    public static void save(Map<String, Tensor> namedParameters, String filePath)
            throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(filePath))) {
            for (Map.Entry<String, Tensor> entry : namedParameters.entrySet()) {
                String name = entry.getKey();
                Tensor tensor = entry.getValue();
                int[] shape = tensor.getShape();

                // Write header: --- name [dim0 dim1 ...]
                StringBuilder header = new StringBuilder("--- ");
                header.append(name).append(" [");
                for (int i = 0; i < shape.length; i++) {
                    if (i > 0) header.append(' ');
                    header.append(shape[i]);
                }
                header.append(']');
                writer.write(header.toString());
                writer.newLine();

                // Write data values as space-separated doubles
                double[] data = tensor.getData();
                StringBuilder values = new StringBuilder();
                for (int i = 0; i < data.length; i++) {
                    if (i > 0) values.append(' ');
                    values.append(data[i]);
                }
                writer.write(values.toString());
                writer.newLine();

                // Blank line between tensors for readability
                writer.newLine();
            }
        }
    }

    /**
     * Loads parameter values from a plain text file into existing tensors.
     *
     * Each tensor in the file is matched by name to the provided map. Values
     * are copied into the existing tensor's data array (the tensor objects
     * themselves are not replaced).
     *
     * @param namedParameters map of parameter names to existing tensors to fill
     * @param filePath        path to the saved model file
     * @throws IOException              if the file cannot be read
     * @throws IllegalArgumentException if a tensor name is not found in the map,
     *                                  or if shapes do not match
     */
    public static void load(Map<String, Tensor> namedParameters, String filePath)
            throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(Paths.get(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();

                // Skip blank lines and comments
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }

                // Parse header line: --- name [dim0 dim1 ...]
                if (!line.startsWith("---")) {
                    throw new IOException("Expected header line starting with '---', got: " + line);
                }

                String headerContent = line.substring(3).trim();
                int bracketStart = headerContent.indexOf('[');
                int bracketEnd = headerContent.indexOf(']');
                if (bracketStart < 0 || bracketEnd < 0) {
                    throw new IOException("Malformed header line (missing brackets): " + line);
                }

                String name = headerContent.substring(0, bracketStart).trim();
                String shapeStr = headerContent.substring(bracketStart + 1, bracketEnd).trim();

                int[] fileShape = parseShape(shapeStr);

                Tensor target = namedParameters.get(name);
                if (target == null) {
                    throw new IllegalArgumentException(
                            "Parameter '" + name + "' from file not found in model");
                }

                // Validate shape match
                int[] targetShape = target.getShape();
                if (!Arrays.equals(fileShape, targetShape)) {
                    throw new IllegalArgumentException(
                            "Shape mismatch for '" + name + "': file has " +
                            Arrays.toString(fileShape) + " but model has " +
                            Arrays.toString(targetShape));
                }

                // Read values line
                String valuesLine = reader.readLine();
                if (valuesLine == null) {
                    throw new IOException("Unexpected end of file after header for '" + name + "'");
                }

                double[] data = target.getData();
                String[] tokens = valuesLine.trim().split("\\s+");
                if (tokens.length != data.length) {
                    throw new IOException(
                            "Value count mismatch for '" + name + "': expected " +
                            data.length + " but got " + tokens.length);
                }

                for (int i = 0; i < data.length; i++) {
                    data[i] = Double.parseDouble(tokens[i]);
                }
            }
        }
    }

    private static int[] parseShape(String shapeStr) {
        if (shapeStr.isEmpty()) {
            return new int[0];
        }
        String[] parts = shapeStr.split("\\s+");
        int[] shape = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            shape[i] = Integer.parseInt(parts[i]);
        }
        return shape;
    }
}
