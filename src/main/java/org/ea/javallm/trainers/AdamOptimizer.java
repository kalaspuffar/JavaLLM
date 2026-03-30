package org.ea.javallm.trainers;

import org.ea.javallm.autograd.Tensor;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Adam optimizer with bias-corrected first and second moment estimates.
 *
 * This optimizer is decoupled from the forward/backward pass: the training loop
 * calls forward, backward, then {@link #step()} to update weights, and
 * {@link #zeroGrad()} to clear gradients before the next iteration.
 *
 * Per-parameter state (first moment m and second moment v) is stored in maps
 * keyed by Tensor identity, matching PyTorch's approach.
 */
public class AdamOptimizer {

    private static final double DEFAULT_LEARNING_RATE = 3e-4;
    private static final double DEFAULT_BETA1 = 0.9;
    private static final double DEFAULT_BETA2 = 0.999;
    private static final double DEFAULT_EPSILON = 1e-8;

    private final List<Tensor> parameters;
    private final double learningRate;
    private final double beta1;
    private final double beta2;
    private final double epsilon;

    private final Map<Tensor, double[]> firstMoment;
    private final Map<Tensor, double[]> secondMoment;
    private int timestep;

    /**
     * Creates an Adam optimizer with default hyperparameters (lr=3e-4, beta1=0.9,
     * beta2=0.999, eps=1e-8).
     *
     * @param parameters list of learnable parameter tensors
     */
    public AdamOptimizer(List<Tensor> parameters) {
        this(parameters, DEFAULT_LEARNING_RATE);
    }

    /**
     * Creates an Adam optimizer with a custom learning rate and default betas/epsilon.
     *
     * @param parameters   list of learnable parameter tensors
     * @param learningRate step size for parameter updates
     */
    public AdamOptimizer(List<Tensor> parameters, double learningRate) {
        this(parameters, learningRate, DEFAULT_BETA1, DEFAULT_BETA2, DEFAULT_EPSILON);
    }

    /**
     * Creates an Adam optimizer with fully specified hyperparameters.
     *
     * @param parameters   list of learnable parameter tensors
     * @param learningRate step size for parameter updates
     * @param beta1        exponential decay rate for the first moment (default 0.9)
     * @param beta2        exponential decay rate for the second moment (default 0.999)
     * @param epsilon      small constant for numerical stability (default 1e-8)
     */
    public AdamOptimizer(List<Tensor> parameters, double learningRate,
                         double beta1, double beta2, double epsilon) {
        this.parameters = parameters;
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.timestep = 0;

        this.firstMoment = new IdentityHashMap<>();
        this.secondMoment = new IdentityHashMap<>();
        for (Tensor param : parameters) {
            int size = param.size();
            firstMoment.put(param, new double[size]);
            secondMoment.put(param, new double[size]);
        }
    }

    /**
     * Performs a single optimization step: reads gradients from each parameter
     * and updates the parameter data using the Adam rule with bias correction.
     */
    public void step() {
        timestep++;
        double biasCorrection1 = 1.0 - Math.pow(beta1, timestep);
        double biasCorrection2 = 1.0 - Math.pow(beta2, timestep);

        for (Tensor param : parameters) {
            double[] data = param.getData();
            double[] grad = param.getGrad();
            if (grad == null) continue;

            double[] m = firstMoment.get(param);
            double[] v = secondMoment.get(param);

            for (int i = 0; i < data.length; i++) {
                double g = grad[i];

                // Update biased first moment estimate
                m[i] = beta1 * m[i] + (1.0 - beta1) * g;

                // Update biased second moment estimate
                v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;

                // Bias-corrected estimates
                double mHat = m[i] / biasCorrection1;
                double vHat = v[i] / biasCorrection2;

                // Update parameter
                data[i] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
            }
        }
    }

    /**
     * Zeros the gradients of all managed parameters. Call this before each
     * forward/backward pass to prevent gradient accumulation across steps.
     */
    public void zeroGrad() {
        for (Tensor param : parameters) {
            param.zeroGrad();
        }
    }
}
