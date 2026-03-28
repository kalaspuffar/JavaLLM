package org.ea.javallm.autograd.ops;

import org.ea.javallm.autograd.Operation;
import org.ea.javallm.autograd.Tensor;

/**
 * Layer normalization across the last dimension.
 *
 * Forward:
 *   x_hat = (x - mean) / sqrt(var + eps)
 *   y = gamma * x_hat + beta
 *
 * Inputs: [input, gamma, beta]
 * where gamma and beta are 1D tensors of size equal to the last dimension.
 *
 * Backward derivation (for each slice of size D along the last dimension):
 *   Let N = D (normalization dimension size)
 *   dx_hat = dout * gamma
 *   dvar = sum(dx_hat * (x - mean) * -0.5 * (var + eps)^(-3/2))
 *   dmean = sum(dx_hat * -1/sqrt(var + eps)) + dvar * sum(-2*(x-mean))/N
 *   dx = dx_hat / sqrt(var + eps) + dvar * 2*(x-mean)/N + dmean/N
 *   dgamma = sum(dout * x_hat)  [over all slices]
 *   dbeta = sum(dout)           [over all slices]
 */
public class LayerNormOp extends Operation {

    private final double eps;
    // Cached for backward pass
    private final double[] mean;
    private final double[] invStd;
    private final double[] xHat;

    public LayerNormOp(Tensor input, Tensor gamma, Tensor beta, Tensor output,
                       double eps, double[] mean, double[] invStd, double[] xHat) {
        super(new Tensor[]{input, gamma, beta}, output);
        this.eps = eps;
        this.mean = mean;
        this.invStd = invStd;
        this.xHat = xHat;
    }

    public static Tensor forward(Tensor input, Tensor gamma, Tensor beta, double eps) {
        int[] shape = input.getShape();
        int lastDim = shape[shape.length - 1];
        int numSlices = input.size() / lastDim;

        double[] data = input.getData();
        double[] gammaData = gamma.getData();
        double[] betaData = beta.getData();
        double[] result = new double[input.size()];
        double[] meanArr = new double[numSlices];
        double[] invStdArr = new double[numSlices];
        double[] xHatArr = new double[input.size()];

        for (int s = 0; s < numSlices; s++) {
            int offset = s * lastDim;

            // Compute mean
            double sum = 0.0;
            for (int i = 0; i < lastDim; i++) {
                sum += data[offset + i];
            }
            double sliceMean = sum / lastDim;
            meanArr[s] = sliceMean;

            // Compute variance
            double varSum = 0.0;
            for (int i = 0; i < lastDim; i++) {
                double diff = data[offset + i] - sliceMean;
                varSum += diff * diff;
            }
            double variance = varSum / lastDim;
            double sliceInvStd = 1.0 / Math.sqrt(variance + eps);
            invStdArr[s] = sliceInvStd;

            // Normalize, scale, and shift
            for (int i = 0; i < lastDim; i++) {
                double normalized = (data[offset + i] - sliceMean) * sliceInvStd;
                xHatArr[offset + i] = normalized;
                result[offset + i] = normalized * gammaData[i] + betaData[i];
            }
        }

        boolean requiresGrad = input.isRequiresGrad() || gamma.isRequiresGrad() || beta.isRequiresGrad();
        Tensor output = new Tensor(result, input.getShape(), requiresGrad);
        new LayerNormOp(input, gamma, beta, output, eps, meanArr, invStdArr, xHatArr);
        return output;
    }

    @Override
    public void backward() {
        Tensor input = inputs[0];
        Tensor gamma = inputs[1];
        Tensor beta = inputs[2];
        double[] dOut = output.getGrad();
        double[] gammaData = gamma.getData();
        double[] inputData = input.getData();

        int[] shape = input.getShape();
        int lastDim = shape[shape.length - 1];
        int numSlices = input.size() / lastDim;

        // Accumulate dgamma and dbeta across all slices
        if (gamma.isRequiresGrad()) {
            double[] dGamma = gamma.getGrad();
            for (int i = 0; i < input.size(); i++) {
                dGamma[i % lastDim] += dOut[i] * xHat[i];
            }
        }

        if (beta.isRequiresGrad()) {
            double[] dBeta = beta.getGrad();
            for (int i = 0; i < input.size(); i++) {
                dBeta[i % lastDim] += dOut[i];
            }
        }

        // Compute gradient for input
        // For each slice:
        //   dx_hat = dout * gamma
        //   dx = (1/N) * invStd * (N * dx_hat - sum(dx_hat) - x_hat * sum(dx_hat * x_hat))
        if (input.isRequiresGrad()) {
            double[] dInput = input.getGrad();

            for (int s = 0; s < numSlices; s++) {
                int offset = s * lastDim;
                double sliceInvStd = invStd[s];

                // dx_hat = dout * gamma
                double sumDxHat = 0.0;
                double sumDxHatXHat = 0.0;
                for (int i = 0; i < lastDim; i++) {
                    double dxHat = dOut[offset + i] * gammaData[i];
                    sumDxHat += dxHat;
                    sumDxHatXHat += dxHat * xHat[offset + i];
                }

                // Simplified backward formula (numerically stable):
                // dx_i = invStd/N * (N * dx_hat_i - sum(dx_hat) - x_hat_i * sum(dx_hat * x_hat))
                double invN = 1.0 / lastDim;
                for (int i = 0; i < lastDim; i++) {
                    double dxHat = dOut[offset + i] * gammaData[i];
                    dInput[offset + i] += sliceInvStd * invN *
                            (lastDim * dxHat - sumDxHat - xHat[offset + i] * sumDxHatXHat);
                }
            }
        }
    }
}
