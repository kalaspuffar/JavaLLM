package org.ea.javallm.autograd.ops;

import org.ea.javallm.autograd.Operation;
import org.ea.javallm.autograd.Tensor;

/**
 * Rectified Linear Unit activation: C = max(0, A)
 *
 * Backward:
 *   dA = dC * (A > 0 ? 1 : 0)
 * Gradient is passed through where input was positive, zeroed where input was negative.
 */
public class ReLU extends Operation {

    public ReLU(Tensor input, Tensor output) {
        super(new Tensor[]{input}, output);
    }

    public static Tensor forward(Tensor input) {
        double[] data = input.getData();
        double[] result = new double[input.size()];

        for (int i = 0; i < result.length; i++) {
            result[i] = Math.max(0.0, data[i]);
        }

        Tensor output = new Tensor(result, input.getShape(), input.isRequiresGrad());
        new ReLU(input, output);
        return output;
    }

    @Override
    public void backward() {
        Tensor input = inputs[0];
        if (!input.isRequiresGrad()) return;

        double[] dInput = input.getGrad();
        double[] dC = output.getGrad();
        double[] inputData = input.getData();

        for (int i = 0; i < dInput.length; i++) {
            // Gradient flows through only where input was strictly positive
            if (inputData[i] > 0) {
                dInput[i] += dC[i];
            }
        }
    }
}
