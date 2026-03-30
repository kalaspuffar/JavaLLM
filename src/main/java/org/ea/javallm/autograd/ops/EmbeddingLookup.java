package org.ea.javallm.autograd.ops;

import org.ea.javallm.autograd.Operation;
import org.ea.javallm.autograd.Tensor;

/**
 * Selects rows from a weight matrix by integer indices.
 *
 * Input: weight matrix of shape (vocabSize, embedDim), indices tensor of shape (seqLen)
 * Output: tensor of shape (seqLen, embedDim) where each row is the embedding for that index.
 *
 * Backward: scatters output gradients back only to the selected rows of the weight gradient.
 * Rows not selected by any index receive zero gradient. If the same index appears multiple
 * times, its gradient contributions are accumulated.
 */
public class EmbeddingLookup extends Operation {

    private final int[] indices;
    private final int embedDim;

    public EmbeddingLookup(Tensor weights, Tensor indicesTensor, Tensor output,
                           int[] indices, int embedDim) {
        super(new Tensor[]{weights, indicesTensor}, output);
        this.indices = indices;
        this.embedDim = embedDim;
    }

    public static Tensor forward(Tensor weights, Tensor indicesTensor) {
        int vocabSize = weights.size(0);
        int embedDim = weights.size(1);
        int seqLen = indicesTensor.size();
        double[] weightData = weights.getData();
        double[] indexData = indicesTensor.getData();

        int[] indices = new int[seqLen];
        double[] result = new double[seqLen * embedDim];

        for (int i = 0; i < seqLen; i++) {
            indices[i] = (int) indexData[i];
            int srcOffset = indices[i] * embedDim;
            int dstOffset = i * embedDim;
            System.arraycopy(weightData, srcOffset, result, dstOffset, embedDim);
        }

        Tensor output = new Tensor(result, new int[]{seqLen, embedDim}, weights.isRequiresGrad());
        new EmbeddingLookup(weights, indicesTensor, output, indices, embedDim);
        return output;
    }

    @Override
    public void backward() {
        Tensor weights = inputs[0];
        if (!weights.isRequiresGrad()) return;

        double[] dWeights = weights.getGrad();
        double[] dOutput = output.getGrad();

        // Scatter: for each index, accumulate the output gradient into the corresponding weight row
        for (int i = 0; i < indices.length; i++) {
            int weightOffset = indices[i] * embedDim;
            int outputOffset = i * embedDim;
            for (int j = 0; j < embedDim; j++) {
                dWeights[weightOffset + j] += dOutput[outputOffset + j];
            }
        }
    }
}
