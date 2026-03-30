package org.ea.javallm.autograd.ops;

import org.ea.javallm.autograd.Operation;
import org.ea.javallm.autograd.Tensor;

/**
 * Numerically stable softmax along a specified dimension.
 *
 * Forward: subtracts the max value along the dimension before exponentiating
 * to prevent overflow: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 *
 * Backward: Jacobian-vector product
 *   dA_i = S_i * (dC_i - sum_j(dC_j * S_j))
 * where S is the softmax output.
 */
public class Softmax extends Operation {

    private final int dim;

    public Softmax(Tensor input, Tensor output, int dim) {
        super(new Tensor[]{input}, output);
        this.dim = dim;
    }

    public static Tensor forward(Tensor input, int dim) {
        if (dim < 0) {
            dim = input.dims() + dim;
        }

        int[] shape = input.getShape();
        double[] data = input.getData();
        double[] result = new double[input.size()];

        // Compute sizes for iterating: outerSize * dimSize * innerSize = total
        int outerSize = 1;
        for (int i = 0; i < dim; i++) {
            outerSize *= shape[i];
        }
        int dimSize = shape[dim];
        int innerSize = 1;
        for (int i = dim + 1; i < shape.length; i++) {
            innerSize *= shape[i];
        }

        for (int outer = 0; outer < outerSize; outer++) {
            for (int inner = 0; inner < innerSize; inner++) {
                int baseIdx = outer * dimSize * innerSize + inner;

                // Find max for numerical stability
                double max = Double.NEGATIVE_INFINITY;
                for (int d = 0; d < dimSize; d++) {
                    double val = data[baseIdx + d * innerSize];
                    if (val > max) max = val;
                }

                // Compute exp(x - max) and sum
                double sumExp = 0.0;
                for (int d = 0; d < dimSize; d++) {
                    int idx = baseIdx + d * innerSize;
                    result[idx] = Math.exp(data[idx] - max);
                    sumExp += result[idx];
                }

                // Normalize
                for (int d = 0; d < dimSize; d++) {
                    result[baseIdx + d * innerSize] /= sumExp;
                }
            }
        }

        Tensor output = new Tensor(result, input.getShape(), input.isRequiresGrad());
        new Softmax(input, output, dim);
        return output;
    }

    @Override
    public void backward() {
        Tensor input = inputs[0];
        if (!input.isRequiresGrad()) return;

        double[] dC = output.getGrad();
        double[] dInput = input.getGrad();
        double[] softmaxOutput = output.getData();

        int[] shape = input.getShape();
        int outerSize = 1;
        for (int i = 0; i < dim; i++) {
            outerSize *= shape[i];
        }
        int dimSize = shape[dim];
        int innerSize = 1;
        for (int i = dim + 1; i < shape.length; i++) {
            innerSize *= shape[i];
        }

        for (int outer = 0; outer < outerSize; outer++) {
            for (int inner = 0; inner < innerSize; inner++) {
                int baseIdx = outer * dimSize * innerSize + inner;

                // Compute dot = sum(dC_j * S_j) along this softmax slice
                double dot = 0.0;
                for (int d = 0; d < dimSize; d++) {
                    int idx = baseIdx + d * innerSize;
                    dot += dC[idx] * softmaxOutput[idx];
                }

                // dInput_i = S_i * (dC_i - dot)
                for (int d = 0; d < dimSize; d++) {
                    int idx = baseIdx + d * innerSize;
                    dInput[idx] += softmaxOutput[idx] * (dC[idx] - dot);
                }
            }
        }
    }
}
