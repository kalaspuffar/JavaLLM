package org.ea.javallm.layers;

import org.ea.javallm.autograd.Tensor;

import java.util.Collections;
import java.util.List;

/**
 * Sinusoidal positional encoding from "Attention Is All You Need" (Section 3.5).
 *
 * Precomputes a table of shape (maxSeqLen, embedDim) at construction time using:
 *   PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
 *   PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
 *
 * The table is stored as a non-gradient tensor. Forward adds the first seqLen
 * rows to the input, injecting position information without learnable parameters.
 */
public class PositionalEncoding {

    private final Tensor encodingTable;
    private final int embedDim;

    /**
     * @param maxSeqLen maximum sequence length supported
     * @param embedDim  embedding dimensionality (must match input last dimension)
     */
    public PositionalEncoding(int maxSeqLen, int embedDim) {
        this.embedDim = embedDim;
        double[] table = new double[maxSeqLen * embedDim];

        for (int pos = 0; pos < maxSeqLen; pos++) {
            for (int i = 0; i < embedDim; i++) {
                // The division index uses integer division: dim pair index = i / 2
                double angle = pos / Math.pow(10000.0, (2.0 * (i / 2)) / embedDim);
                if (i % 2 == 0) {
                    table[pos * embedDim + i] = Math.sin(angle);
                } else {
                    table[pos * embedDim + i] = Math.cos(angle);
                }
            }
        }

        this.encodingTable = new Tensor(table, new int[]{maxSeqLen, embedDim}, false);
        this.encodingTable.setName("PositionalEncoding.table");
    }

    /**
     * Adds positional encoding to the input.
     *
     * @param input tensor of shape (batch, seqLen, embedDim)
     * @return tensor of shape (batch, seqLen, embedDim) with position info added
     */
    public Tensor forward(Tensor input) {
        int batch = input.size(0);
        int seqLen = input.size(1);

        // The Add op only broadcasts 1D bias vectors, so we tile the PE slice
        // across the batch dimension to match the input shape (batch, seqLen, embedDim).
        int sliceSize = seqLen * embedDim;
        double[] tiledData = new double[batch * sliceSize];
        double[] tableData = encodingTable.getData();
        for (int b = 0; b < batch; b++) {
            System.arraycopy(tableData, 0, tiledData, b * sliceSize, sliceSize);
        }
        Tensor posEncoding = new Tensor(tiledData, new int[]{batch, seqLen, embedDim}, false);

        return input.add(posEncoding);
    }

    /**
     * No learnable parameters — returns an empty list.
     */
    public List<Tensor> getParameters() {
        return Collections.emptyList();
    }

    /**
     * Returns the full precomputed encoding table for inspection/testing.
     */
    public Tensor getEncodingTable() {
        return encodingTable;
    }
}
