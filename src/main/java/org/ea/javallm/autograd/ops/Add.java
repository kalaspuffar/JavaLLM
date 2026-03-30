package org.ea.javallm.autograd.ops;

import org.ea.javallm.autograd.Operation;
import org.ea.javallm.autograd.Tensor;

/**
 * Element-wise addition with broadcasting support: C = A + B
 *
 * Supports broadcasting a (n) or (1,n) bias tensor to match a (batch, seq, n) tensor.
 * When shapes match exactly, performs simple element-wise addition.
 *
 * Backward:
 *   dA = dC (identity)
 *   dB = dC summed over broadcast dimensions (if B was broadcast)
 */
public class Add extends Operation {

    public Add(Tensor a, Tensor b, Tensor output) {
        super(new Tensor[]{a, b}, output);
    }

    public static Tensor forward(Tensor a, Tensor b) {
        double[] aData = a.getData();
        double[] bData = b.getData();

        if (a.size() == b.size() && java.util.Arrays.equals(a.getShape(), b.getShape())) {
            // Same-shape addition: no broadcasting needed
            return forwardSameShape(a, b);
        } else {
            // Broadcasting: B is smaller and gets broadcast to A's shape
            return forwardBroadcast(a, b);
        }
    }

    private static Tensor forwardSameShape(Tensor a, Tensor b) {
        double[] aData = a.getData();
        double[] bData = b.getData();
        double[] result = new double[a.size()];

        for (int i = 0; i < result.length; i++) {
            result[i] = aData[i] + bData[i];
        }

        boolean requiresGrad = a.isRequiresGrad() || b.isRequiresGrad();
        Tensor output = new Tensor(result, a.getShape(), requiresGrad);
        new Add(a, b, output);
        return output;
    }

    private static Tensor forwardBroadcast(Tensor a, Tensor b) {
        double[] aData = a.getData();
        double[] bData = b.getData();
        int lastDimA = a.size(a.dims() - 1);

        // B must be (n) or (1,n) where n matches A's last dimension
        int lastDimB = b.size(b.dims() - 1);
        if (lastDimB != lastDimA || b.size() != lastDimB) {
            throw new IllegalArgumentException(
                    "Add broadcast requires B to be (" + lastDimA + ") or (1," + lastDimA +
                    "), but B has " + b.size() + " elements");
        }

        double[] result = new double[a.size()];
        for (int i = 0; i < a.size(); i++) {
            result[i] = aData[i] + bData[i % lastDimA];
        }

        boolean requiresGrad = a.isRequiresGrad() || b.isRequiresGrad();
        Tensor output = new Tensor(result, a.getShape(), requiresGrad);
        new Add(a, b, output);
        return output;
    }

    @Override
    public void backward() {
        Tensor a = inputs[0];
        Tensor b = inputs[1];
        double[] dC = output.getGrad();

        // dA = dC (identity, accumulated)
        if (a.isRequiresGrad()) {
            double[] dA = a.getGrad();
            for (int i = 0; i < dA.length; i++) {
                dA[i] += dC[i];
            }
        }

        // dB = dC summed over broadcast dimensions if B was broadcast
        if (b.isRequiresGrad()) {
            double[] dB = b.getGrad();
            if (b.size() == output.size()) {
                // Same shape: identity gradient
                for (int i = 0; i < dB.length; i++) {
                    dB[i] += dC[i];
                }
            } else {
                // Broadcast case: sum over all positions that mapped to each B element
                int lastDim = b.size();
                for (int i = 0; i < dC.length; i++) {
                    dB[i % lastDim] += dC[i];
                }
            }
        }
    }
}
