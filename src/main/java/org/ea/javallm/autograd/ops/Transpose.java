package org.ea.javallm.autograd.ops;

import org.ea.javallm.autograd.Operation;
import org.ea.javallm.autograd.Tensor;

/**
 * Transposes two dimensions of a tensor with a physical data copy.
 *
 * For a 2D tensor, this is the standard matrix transpose.
 * For higher-dimensional tensors, it swaps two specified dimensions.
 *
 * Backward: transpose the gradient back using the same dimension swap
 * (transposing the same dims twice restores the original layout).
 */
public class Transpose extends Operation {

    private final int dim0;
    private final int dim1;

    public Transpose(Tensor input, Tensor output, int dim0, int dim1) {
        super(new Tensor[]{input}, output);
        this.dim0 = dim0;
        this.dim1 = dim1;
    }

    public static Tensor forward(Tensor input, int dim0, int dim1) {
        int[] oldShape = input.getShape();
        int ndims = oldShape.length;

        if (dim0 < 0 || dim0 >= ndims || dim1 < 0 || dim1 >= ndims) {
            throw new IllegalArgumentException(
                    "Transpose dims (" + dim0 + "," + dim1 + ") out of range for " + ndims + "D tensor");
        }

        // Build new shape by swapping the two dimensions
        int[] newShape = oldShape.clone();
        newShape[dim0] = oldShape[dim1];
        newShape[dim1] = oldShape[dim0];

        // Build the permutation: identity with dim0 and dim1 swapped
        int[] perm = new int[ndims];
        for (int i = 0; i < ndims; i++) perm[i] = i;
        perm[dim0] = dim1;
        perm[dim1] = dim0;

        double[] result = transposeData(input.getData(), oldShape, newShape, perm);

        Tensor output = new Tensor(result, newShape, input.isRequiresGrad());
        new Transpose(input, output, dim0, dim1);
        return output;
    }

    /**
     * Copies data from source layout to destination layout according to a dimension permutation.
     * For each multi-dimensional index in the output, the corresponding source index
     * is obtained by applying the inverse permutation (which for a swap is the same swap).
     */
    private static double[] transposeData(double[] srcData, int[] srcShape, int[] dstShape, int[] perm) {
        int ndims = srcShape.length;
        int totalElements = srcData.length;
        double[] dstData = new double[totalElements];

        // Precompute source and destination strides
        int[] srcStrides = computeStrides(srcShape);
        int[] dstStrides = computeStrides(dstShape);

        // Iterate over all elements using flat index, decompose into dst multi-index,
        // permute to get src multi-index, compute src flat offset
        int[] dstIndices = new int[ndims];

        for (int flatDst = 0; flatDst < totalElements; flatDst++) {
            // Decompose flat dst index into multi-dimensional indices
            int remaining = flatDst;
            for (int d = 0; d < ndims; d++) {
                dstIndices[d] = remaining / dstStrides[d];
                remaining %= dstStrides[d];
            }

            // Map dst indices back to src indices via the permutation
            int flatSrc = 0;
            for (int d = 0; d < ndims; d++) {
                flatSrc += dstIndices[d] * srcStrides[perm[d]];
            }

            dstData[flatDst] = srcData[flatSrc];
        }

        return dstData;
    }

    private static int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        if (shape.length > 0) {
            strides[shape.length - 1] = 1;
            for (int i = shape.length - 2; i >= 0; i--) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        return strides;
    }

    @Override
    public void backward() {
        Tensor input = inputs[0];
        if (!input.isRequiresGrad()) return;

        // Transposing the same two dimensions reverses the permutation
        int[] outShape = output.getShape();
        int[] inShape = input.getShape();
        int ndims = inShape.length;

        int[] perm = new int[ndims];
        for (int i = 0; i < ndims; i++) perm[i] = i;
        perm[dim0] = dim1;
        perm[dim1] = dim0;

        double[] transposedGrad = transposeData(output.getGrad(), outShape, inShape, perm);

        double[] dInput = input.getGrad();
        for (int i = 0; i < dInput.length; i++) {
            dInput[i] += transposedGrad[i];
        }
    }
}
