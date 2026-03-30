package org.ea.javallm.layers;

import org.ea.javallm.autograd.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Position-wise feed-forward network: Linear → ReLU → Linear.
 *
 * This is the standard FFN block from "Attention Is All You Need" (Section 3.3):
 *   FFN(x) = Linear2(ReLU(Linear1(x)))
 *
 * The inner dimension (typically 4 × embedDim) creates a bottleneck that lets
 * the network learn non-linear transformations of each position independently.
 *
 * Note: Modern transformers often use GELU instead of ReLU. ReLU is used here
 * for simplicity and compatibility with the existing autograd operations.
 */
public class FeedForward {

    private final Linear linear1;
    private final Linear linear2;

    /**
     * @param embedDim dimension of input and output features
     * @param innerDim dimension of the hidden layer (typically 4 × embedDim)
     * @param rng      random number generator for weight initialization
     */
    public FeedForward(int embedDim, int innerDim, Random rng) {
        this.linear1 = new Linear(embedDim, innerDim, rng);
        this.linear2 = new Linear(innerDim, embedDim, rng);
    }

    /**
     * Applies the feed-forward network: Linear → ReLU → Linear.
     *
     * @param input tensor of shape (..., embedDim)
     * @return tensor of shape (..., embedDim)
     */
    public Tensor forward(Tensor input) {
        Tensor hidden = linear1.forward(input).relu();
        return linear2.forward(hidden);
    }

    /**
     * Returns learnable parameters from both linear layers for optimizer iteration.
     */
    public List<Tensor> getParameters() {
        List<Tensor> params = new ArrayList<>();
        params.addAll(linear1.getParameters());
        params.addAll(linear2.getParameters());
        return params;
    }
}
