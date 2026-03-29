package org.ea.javallm.layers;

import org.ea.javallm.autograd.Tensor;

import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Token embedding layer that maps integer token IDs to dense vectors.
 *
 * Stores a weight matrix of shape (vocabSize, embedDim) and uses embeddingLookup
 * to select rows by token indices. The weight field is public to enable weight
 * tying with an output projection layer.
 */
public class Embedding {

    /**
     * Embedding weight matrix of shape (vocabSize, embedDim).
     * Public to allow weight tying with the output projection.
     */
    public final Tensor weight;

    /**
     * @param vocabSize number of tokens in the vocabulary
     * @param embedDim  dimensionality of each token embedding
     * @param rng       random number generator for weight initialization
     */
    public Embedding(int vocabSize, int embedDim, Random rng) {
        int size = vocabSize * embedDim;
        double[] data = new double[size];
        double scale = 1.0 / Math.sqrt(embedDim);
        for (int i = 0; i < size; i++) {
            data[i] = rng.nextGaussian() * scale;
        }
        this.weight = new Tensor(data, new int[]{vocabSize, embedDim}, true);
        this.weight.setName("Embedding.weight");
    }

    /**
     * Looks up embeddings for the given token indices.
     *
     * EmbeddingLookup treats indices as a flat sequence, so for batched inputs
     * with shape (batch, seqLen) we reshape the output from (batch*seqLen, embedDim)
     * back to (batch, seqLen, embedDim).
     *
     * @param indices tensor of integer token IDs with shape (seqLen) or (batch, seqLen)
     * @return tensor of shape (seqLen, embedDim) or (batch, seqLen, embedDim)
     */
    public Tensor forward(Tensor indices) {
        int embedDim = weight.size(1);
        Tensor flat = weight.embeddingLookup(indices);

        if (indices.dims() == 2) {
            int batch = indices.size(0);
            int seqLen = indices.size(1);
            return flat.reshape(batch, seqLen, embedDim);
        }
        return flat;
    }

    /**
     * Returns the learnable parameters (the weight matrix) for optimizer iteration.
     */
    public List<Tensor> getParameters() {
        return Collections.singletonList(weight);
    }
}
