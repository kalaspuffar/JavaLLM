package org.ea.javallm.autograd.ops;

import org.ea.javallm.autograd.Operation;
import org.ea.javallm.autograd.Tensor;

/**
 * Applies a fill value at masked positions.
 *
 * Positions where the mask is 0 are replaced with fillValue (typically -1e9 for
 * pre-softmax attention masking). Other positions pass through unchanged.
 *
 * Backward: gradient is zeroed at masked positions (since those positions are
 * constant values, not functions of the input).
 */
public class Mask extends Operation {

    private final double[] maskData;
    private final double fillValue;

    public Mask(Tensor input, Tensor maskTensor, Tensor output, double fillValue) {
        super(new Tensor[]{input, maskTensor}, output);
        this.maskData = maskTensor.getData();
        this.fillValue = fillValue;
    }

    public static Tensor forward(Tensor input, Tensor maskTensor, double fillValue) {
        if (input.size() != maskTensor.size()) {
            throw new IllegalArgumentException(
                    "Mask and input must have the same size, got " + input.size() +
                    " and " + maskTensor.size());
        }

        double[] data = input.getData();
        double[] mask = maskTensor.getData();
        double[] result = new double[input.size()];

        for (int i = 0; i < result.length; i++) {
            // mask value of 0 means "masked out" (fill), non-zero means "keep"
            result[i] = (mask[i] == 0.0) ? fillValue : data[i];
        }

        Tensor output = new Tensor(result, input.getShape(), input.isRequiresGrad());
        new Mask(input, maskTensor, output, fillValue);
        return output;
    }

    @Override
    public void backward() {
        Tensor input = inputs[0];
        if (!input.isRequiresGrad()) return;

        double[] dInput = input.getGrad();
        double[] dOutput = output.getGrad();

        for (int i = 0; i < dInput.length; i++) {
            // Gradient flows through only at non-masked positions
            if (maskData[i] != 0.0) {
                dInput[i] += dOutput[i];
            }
        }
    }
}
