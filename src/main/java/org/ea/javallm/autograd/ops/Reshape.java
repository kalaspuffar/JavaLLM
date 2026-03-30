package org.ea.javallm.autograd.ops;

import org.ea.javallm.autograd.Operation;
import org.ea.javallm.autograd.Tensor;

import java.util.Arrays;

/**
 * Reshapes a tensor to a new shape without altering the underlying data.
 *
 * The total number of elements must match between old and new shapes.
 * Since data is stored in flat row-major order, reshape only changes
 * how the shape/strides are interpreted — the data array is copied as-is.
 *
 * Backward: reshape the gradient back to the original (input) shape.
 */
public class Reshape extends Operation {

    private final int[] originalShape;

    public Reshape(Tensor input, Tensor output, int[] originalShape) {
        super(new Tensor[]{input}, output);
        this.originalShape = originalShape;
    }

    public static Tensor forward(Tensor input, int... newShape) {
        int oldSize = input.size();
        int newSize = 1;
        for (int dim : newShape) {
            newSize *= dim;
        }

        if (oldSize != newSize) {
            throw new IllegalArgumentException(
                    "Cannot reshape tensor of size " + oldSize + " to shape " +
                    Arrays.toString(newShape) + " (size " + newSize + ")");
        }

        // Copy data — same flat layout, different shape interpretation
        double[] newData = input.getData().clone();
        Tensor output = new Tensor(newData, newShape, input.isRequiresGrad());
        new Reshape(input, output, input.getShape());
        return output;
    }

    @Override
    public void backward() {
        Tensor input = inputs[0];
        if (!input.isRequiresGrad()) return;

        double[] dOutput = output.getGrad();
        double[] dInput = input.getGrad();

        // Gradient flows straight through — same flat data, just different shape
        for (int i = 0; i < dInput.length; i++) {
            dInput[i] += dOutput[i];
        }
    }
}
