package org.ea.javallm.layers;

import org.ea.javallm.autograd.Tensor;

import java.util.Arrays;
import java.util.List;

/**
 * Layer normalization with learnable scale (gamma) and shift (beta).
 *
 * Normalizes each feature vector along the last dimension to zero mean and
 * unit variance, then applies a learnable affine transformation:
 *   output = gamma * normalize(input) + beta
 *
 * Gamma is initialized to 1.0 and beta to 0.0, so the initial behavior is
 * a simple normalization pass-through.
 */
public class LayerNorm {

    private final Tensor gamma;
    private final Tensor beta;
    private final double eps;

    /**
     * @param embedDim size of the last dimension to normalize over
     */
    public LayerNorm(int embedDim) {
        this(embedDim, 1e-5);
    }

    /**
     * @param embedDim size of the last dimension to normalize over
     * @param eps      small constant for numerical stability in variance computation
     */
    public LayerNorm(int embedDim, double eps) {
        this.eps = eps;

        double[] gammaData = new double[embedDim];
        java.util.Arrays.fill(gammaData, 1.0);
        this.gamma = new Tensor(gammaData, new int[]{embedDim}, true);
        this.gamma.setName("LayerNorm.gamma");

        this.beta = new Tensor(new double[embedDim], new int[]{embedDim}, true);
        this.beta.setName("LayerNorm.beta");
    }

    /**
     * Applies layer normalization to the input.
     *
     * @param input tensor of shape (..., embedDim)
     * @return normalized tensor with the same shape
     */
    public Tensor forward(Tensor input) {
        return input.layerNorm(gamma, beta, eps);
    }

    /**
     * Returns the learnable parameters (gamma and beta) for optimizer iteration.
     */
    public List<Tensor> getParameters() {
        return Arrays.asList(gamma, beta);
    }

    public Tensor getGamma() {
        return gamma;
    }

    public Tensor getBeta() {
        return beta;
    }
}
