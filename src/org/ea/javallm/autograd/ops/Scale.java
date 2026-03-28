package org.ea.javallm.autograd.ops;

import org.ea.javallm.autograd.Operation;
import org.ea.javallm.autograd.Tensor;

/**
 * Scalar scaling: C = A * scalar
 *
 * Backward:
 *   dA = dC * scalar
 */
public class Scale extends Operation {

    private final double scalar;

    public Scale(Tensor input, Tensor output, double scalar) {
        super(new Tensor[]{input}, output);
        this.scalar = scalar;
    }

    public static Tensor forward(Tensor input, double scalar) {
        double[] data = input.getData();
        double[] result = new double[input.size()];

        for (int i = 0; i < result.length; i++) {
            result[i] = data[i] * scalar;
        }

        Tensor output = new Tensor(result, input.getShape(), input.isRequiresGrad());
        new Scale(input, output, scalar);
        return output;
    }

    @Override
    public void backward() {
        Tensor input = inputs[0];
        if (input.isRequiresGrad()) {
            double[] dInput = input.getGrad();
            double[] dC = output.getGrad();
            for (int i = 0; i < dInput.length; i++) {
                dInput[i] += dC[i] * scalar;
            }
        }
    }
}
