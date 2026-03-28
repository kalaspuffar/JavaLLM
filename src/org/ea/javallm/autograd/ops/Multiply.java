package org.ea.javallm.autograd.ops;

import org.ea.javallm.autograd.Operation;
import org.ea.javallm.autograd.Tensor;

/**
 * Element-wise Hadamard product: C = A * B
 *
 * Both tensors must have the same shape (no broadcasting).
 *
 * Backward:
 *   dA = dC * B
 *   dB = dC * A
 */
public class Multiply extends Operation {

    public Multiply(Tensor a, Tensor b, Tensor output) {
        super(new Tensor[]{a, b}, output);
    }

    public static Tensor forward(Tensor a, Tensor b) {
        if (a.size() != b.size()) {
            throw new IllegalArgumentException(
                    "Multiply requires matching sizes, got " + a.size() + " and " + b.size());
        }

        double[] aData = a.getData();
        double[] bData = b.getData();
        double[] result = new double[a.size()];

        for (int i = 0; i < result.length; i++) {
            result[i] = aData[i] * bData[i];
        }

        boolean requiresGrad = a.isRequiresGrad() || b.isRequiresGrad();
        Tensor output = new Tensor(result, a.getShape(), requiresGrad);
        new Multiply(a, b, output);
        return output;
    }

    @Override
    public void backward() {
        Tensor a = inputs[0];
        Tensor b = inputs[1];
        double[] dC = output.getGrad();

        if (a.isRequiresGrad()) {
            double[] dA = a.getGrad();
            double[] bData = b.getData();
            for (int i = 0; i < dA.length; i++) {
                dA[i] += dC[i] * bData[i];
            }
        }

        if (b.isRequiresGrad()) {
            double[] dB = b.getGrad();
            double[] aData = a.getData();
            for (int i = 0; i < dB.length; i++) {
                dB[i] += dC[i] * aData[i];
            }
        }
    }
}
