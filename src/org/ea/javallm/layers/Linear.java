package org.ea.javallm.layers;

import org.ea.javallm.autograd.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Linear projection layer: output = input @ weight^T + bias.
 *
 * Weight shape is (outputDim, inputDim) following PyTorch convention, where each
 * row corresponds to one output neuron. Forward transposes the weight matrix so
 * that a batched input of shape (..., inputDim) produces output of shape (..., outputDim).
 *
 * Weights are initialized with Xavier/Glorot scaling: N(0, sqrt(2/(fan_in + fan_out))).
 * Biases are initialized to zero.
 */
public class Linear {

    private final Tensor weight;
    private final Tensor bias;

    /**
     * @param inputDim  number of input features
     * @param outputDim number of output features
     * @param rng       random number generator for weight initialization
     */
    public Linear(int inputDim, int outputDim, Random rng) {
        double xavierScale = Math.sqrt(2.0 / (inputDim + outputDim));
        int weightSize = outputDim * inputDim;
        double[] weightData = new double[weightSize];
        for (int i = 0; i < weightSize; i++) {
            weightData[i] = rng.nextGaussian() * xavierScale;
        }
        this.weight = new Tensor(weightData, new int[]{outputDim, inputDim}, true);
        this.weight.setName("Linear.weight");

        this.bias = new Tensor(new double[outputDim], new int[]{outputDim}, true);
        this.bias.setName("Linear.bias");
    }

    /**
     * Computes the linear projection: input @ weight^T + bias.
     *
     * For inputs with more than 2 dimensions (e.g., batch, seqLen, inputDim),
     * the leading dimensions are folded into a single batch dimension for the
     * matmul, then unfolded back to the original shape.
     *
     * @param input tensor of shape (..., inputDim)
     * @return tensor of shape (..., outputDim)
     */
    public Tensor forward(Tensor input) {
        int inputDim = weight.size(1);
        int outputDim = weight.size(0);
        Tensor weightTransposed = weight.transpose(0, 1);

        if (input.dims() == 2) {
            Tensor projected = input.matmul(weightTransposed);
            return projected.add(bias);
        }

        // For N-D input (N > 2), flatten leading dimensions into a single batch dimension,
        // perform the 2D matmul, then restore the original leading dimensions.
        int[] inputShape = input.getShape();
        int leadingSize = 1;
        for (int i = 0; i < inputShape.length - 1; i++) {
            leadingSize *= inputShape[i];
        }

        Tensor flat = input.reshape(leadingSize, inputDim);
        Tensor projected = flat.matmul(weightTransposed);
        Tensor withBias = projected.add(bias);

        // Restore original leading dimensions with outputDim as the last dimension
        int[] outputShape = new int[inputShape.length];
        System.arraycopy(inputShape, 0, outputShape, 0, inputShape.length - 1);
        outputShape[inputShape.length - 1] = outputDim;
        return withBias.reshape(outputShape);
    }

    /**
     * Returns the learnable parameters (weight and bias) for optimizer iteration.
     */
    public List<Tensor> getParameters() {
        return Arrays.asList(weight, bias);
    }

    public Tensor getWeight() {
        return weight;
    }

    public Tensor getBias() {
        return bias;
    }
}
