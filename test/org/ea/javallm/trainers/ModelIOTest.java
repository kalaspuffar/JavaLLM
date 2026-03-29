package org.ea.javallm.trainers;

import org.ea.javallm.autograd.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ModelIOTest {

    @TempDir
    Path tempDir;

    @Test
    void saveLoadRoundTripPreservesValues() throws IOException {
        String filePath = tempDir.resolve("model.txt").toString();

        // Create named parameters with known values
        Tensor weight = new Tensor(
                new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                new int[]{2, 3}, false);
        Tensor bias = new Tensor(
                new double[]{0.1, 0.2, 0.3},
                new int[]{3}, false);

        Map<String, Tensor> saveParams = new LinkedHashMap<>();
        saveParams.put("layer.weight", weight);
        saveParams.put("layer.bias", bias);

        ModelIO.save(saveParams, filePath);

        // Create fresh tensors with same shapes but zeroed data
        Tensor loadWeight = Tensor.zeros(2, 3);
        Tensor loadBias = Tensor.zeros(3);

        Map<String, Tensor> loadParams = new LinkedHashMap<>();
        loadParams.put("layer.weight", loadWeight);
        loadParams.put("layer.bias", loadBias);

        ModelIO.load(loadParams, filePath);

        // Verify values match
        assertArrayEquals(weight.getData(), loadWeight.getData(), 1e-15,
                "Loaded weight values should match saved values");
        assertArrayEquals(bias.getData(), loadBias.getData(), 1e-15,
                "Loaded bias values should match saved values");
    }

    @Test
    void shapeMismatchThrowsException() throws IOException {
        String filePath = tempDir.resolve("model.txt").toString();

        Tensor weight = new Tensor(
                new double[]{1.0, 2.0, 3.0, 4.0},
                new int[]{2, 2}, false);

        Map<String, Tensor> saveParams = new LinkedHashMap<>();
        saveParams.put("w", weight);
        ModelIO.save(saveParams, filePath);

        // Try to load into a tensor with a different shape
        Tensor wrongShape = Tensor.zeros(4, 1);
        Map<String, Tensor> loadParams = new LinkedHashMap<>();
        loadParams.put("w", wrongShape);

        assertThrows(IllegalArgumentException.class,
                () -> ModelIO.load(loadParams, filePath),
                "Loading into a tensor with different shape should throw");
    }

    @Test
    void unknownParameterNameThrowsException() throws IOException {
        String filePath = tempDir.resolve("model.txt").toString();

        Tensor t = new Tensor(new double[]{1.0}, new int[]{1}, false);
        Map<String, Tensor> saveParams = new LinkedHashMap<>();
        saveParams.put("unknown.param", t);
        ModelIO.save(saveParams, filePath);

        // Load with a different parameter name
        Map<String, Tensor> loadParams = new LinkedHashMap<>();
        loadParams.put("different.param", Tensor.zeros(1));

        assertThrows(IllegalArgumentException.class,
                () -> ModelIO.load(loadParams, filePath));
    }
}
