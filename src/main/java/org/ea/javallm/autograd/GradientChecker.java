package org.ea.javallm.autograd;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * Validates autograd gradients against numerical finite-difference approximations.
 *
 * For each element of each parameter tensor, perturbs the value by +/- epsilon,
 * recomputes the forward pass, and estimates the gradient as:
 *   numerical_grad = (loss(x + eps) - loss(x - eps)) / (2 * eps)
 *
 * Compares against the autograd gradient and reports any element where the
 * relative error exceeds the threshold.
 */
public class GradientChecker {

    private final double epsilon;
    private final double threshold;

    public GradientChecker(double epsilon, double threshold) {
        this.epsilon = epsilon;
        this.threshold = threshold;
    }

    /**
     * Creates a GradientChecker with default settings (epsilon=1e-5, threshold=1e-4).
     */
    public GradientChecker() {
        this(1e-5, 1e-4);
    }

    /**
     * Checks gradients for the given parameters.
     *
     * @param forwardPass a supplier that runs the forward pass and returns the scalar loss tensor.
     *                    This will be called multiple times with perturbed parameter values.
     * @param parameters  the tensors whose gradients should be verified. All must have requiresGrad=true.
     * @return a result object with pass/fail status and any failures
     */
    public Result check(Supplier<Tensor> forwardPass, Tensor... parameters) {
        // Step 1: Run forward + backward to get autograd gradients
        for (Tensor param : parameters) {
            param.zeroGrad();
        }
        Tensor loss = forwardPass.get();
        loss.backward();

        // Step 2: For each parameter element, compute numerical gradient
        List<Failure> failures = new ArrayList<>();

        for (int p = 0; p < parameters.length; p++) {
            Tensor param = parameters[p];
            double[] data = param.getData();
            double[] analyticGrad = param.getGrad();

            for (int i = 0; i < data.length; i++) {
                double originalValue = data[i];

                // Forward with +epsilon
                data[i] = originalValue + epsilon;
                Tensor lossPlus = forwardPass.get();
                double fPlus = lossPlus.getData()[0];

                // Forward with -epsilon
                data[i] = originalValue - epsilon;
                Tensor lossMinus = forwardPass.get();
                double fMinus = lossMinus.getData()[0];

                // Restore original value
                data[i] = originalValue;

                double numericalGrad = (fPlus - fMinus) / (2.0 * epsilon);
                double analyticValue = analyticGrad[i];

                double relError = relativeError(numericalGrad, analyticValue);
                if (relError > threshold) {
                    failures.add(new Failure(
                            p, i, numericalGrad, analyticValue, relError,
                            param.getName() != null ? param.getName() : "param[" + p + "]"));
                }
            }
        }

        return new Result(failures);
    }

    /**
     * Computes relative error handling the case where both values are near zero.
     * Uses max(|a|, |b|, 1) in the denominator to avoid division by zero.
     */
    private static double relativeError(double numerical, double analytic) {
        double diff = Math.abs(numerical - analytic);
        double denom = Math.max(Math.max(Math.abs(numerical), Math.abs(analytic)), 1e-8);
        return diff / denom;
    }

    /** A single gradient mismatch. */
    public static class Failure {
        public final int paramIndex;
        public final int elementIndex;
        public final double numericalGrad;
        public final double analyticGrad;
        public final double relativeError;
        public final String paramName;

        public Failure(int paramIndex, int elementIndex, double numericalGrad,
                       double analyticGrad, double relativeError, String paramName) {
            this.paramIndex = paramIndex;
            this.elementIndex = elementIndex;
            this.numericalGrad = numericalGrad;
            this.analyticGrad = analyticGrad;
            this.relativeError = relativeError;
            this.paramName = paramName;
        }

        @Override
        public String toString() {
            return String.format("%s[%d]: numerical=%.8f analytic=%.8f relError=%.6f",
                    paramName, elementIndex, numericalGrad, analyticGrad, relativeError);
        }
    }

    /** The result of a gradient check. */
    public static class Result {
        public final List<Failure> failures;

        public Result(List<Failure> failures) {
            this.failures = failures;
        }

        public boolean passed() {
            return failures.isEmpty();
        }

        @Override
        public String toString() {
            if (passed()) return "GradientCheck PASSED";
            StringBuilder sb = new StringBuilder("GradientCheck FAILED with " + failures.size() + " failures:\n");
            for (Failure f : failures) {
                sb.append("  ").append(f).append("\n");
            }
            return sb.toString();
        }
    }
}
