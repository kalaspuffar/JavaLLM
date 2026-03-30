package org.ea.javallm.autograd.ops;

import org.ea.javallm.autograd.Operation;
import org.ea.javallm.autograd.Tensor;

/**
 * Matrix multiplication operation: C = A @ B
 *
 * Supports both 2D (m,k) @ (k,n) -> (m,n) and batched 3D
 * (b,m,k) @ (b,k,n) -> (b,m,n) where each batch is multiplied independently.
 *
 * Backward:
 *   dA = dC @ B^T
 *   dB = A^T @ dC
 */
public class MatMul extends Operation {

    public MatMul(Tensor a, Tensor b, Tensor output) {
        super(new Tensor[]{a, b}, output);
    }

    /**
     * Computes the forward pass and returns the output tensor.
     */
    public static Tensor forward(Tensor a, Tensor b) {
        int aDims = a.dims();
        int bDims = b.dims();

        if (aDims == 2 && bDims == 2) {
            return forward2D(a, b);
        } else if (aDims == 3 && bDims == 3) {
            return forward3D(a, b);
        } else {
            throw new IllegalArgumentException(
                    "MatMul supports 2D@2D or 3D@3D, got " + aDims + "D @ " + bDims + "D");
        }
    }

    private static Tensor forward2D(Tensor a, Tensor b) {
        int m = a.size(0);
        int k = a.size(1);
        int n = b.size(1);

        if (b.size(0) != k) {
            throw new IllegalArgumentException(
                    "MatMul shape mismatch: (" + m + "," + k + ") @ (" + b.size(0) + "," + n + ")");
        }

        double[] result = new double[m * n];
        double[] aData = a.getData();
        double[] bData = b.getData();

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int p = 0; p < k; p++) {
                    sum += aData[i * k + p] * bData[p * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        boolean requiresGrad = a.isRequiresGrad() || b.isRequiresGrad();
        Tensor output = new Tensor(result, new int[]{m, n}, requiresGrad);
        new MatMul(a, b, output);
        return output;
    }

    private static Tensor forward3D(Tensor a, Tensor b) {
        int batch = a.size(0);
        int m = a.size(1);
        int k = a.size(2);
        int n = b.size(2);

        if (b.size(0) != batch || b.size(1) != k) {
            throw new IllegalArgumentException(
                    "MatMul 3D shape mismatch: " + shapeStr(a) + " @ " + shapeStr(b));
        }

        double[] result = new double[batch * m * n];
        double[] aData = a.getData();
        double[] bData = b.getData();

        for (int bIdx = 0; bIdx < batch; bIdx++) {
            int aOffset = bIdx * m * k;
            int bOffset = bIdx * k * n;
            int cOffset = bIdx * m * n;

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double sum = 0.0;
                    for (int p = 0; p < k; p++) {
                        sum += aData[aOffset + i * k + p] * bData[bOffset + p * n + j];
                    }
                    result[cOffset + i * n + j] = sum;
                }
            }
        }

        boolean requiresGrad = a.isRequiresGrad() || b.isRequiresGrad();
        Tensor output = new Tensor(result, new int[]{batch, m, n}, requiresGrad);
        new MatMul(a, b, output);
        return output;
    }

    @Override
    public void backward() {
        Tensor a = inputs[0];
        Tensor b = inputs[1];
        double[] dC = output.getGrad();

        if (a.dims() == 2) {
            backward2D(a, b, dC);
        } else {
            backward3D(a, b, dC);
        }
    }

    private void backward2D(Tensor a, Tensor b, double[] dC) {
        int m = a.size(0);
        int k = a.size(1);
        int n = b.size(1);
        double[] aData = a.getData();
        double[] bData = b.getData();

        // dA = dC @ B^T: (m,n) @ (n,k) -> (m,k)
        if (a.isRequiresGrad()) {
            double[] dA = a.getGrad();
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < k; j++) {
                    double sum = 0.0;
                    for (int p = 0; p < n; p++) {
                        sum += dC[i * n + p] * bData[j * n + p];
                    }
                    dA[i * k + j] += sum;
                }
            }
        }

        // dB = A^T @ dC: (k,m) @ (m,n) -> (k,n)
        if (b.isRequiresGrad()) {
            double[] dB = b.getGrad();
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < n; j++) {
                    double sum = 0.0;
                    for (int p = 0; p < m; p++) {
                        sum += aData[p * k + i] * dC[p * n + j];
                    }
                    dB[i * n + j] += sum;
                }
            }
        }
    }

    private void backward3D(Tensor a, Tensor b, double[] dC) {
        int batch = a.size(0);
        int m = a.size(1);
        int k = a.size(2);
        int n = b.size(2);
        double[] aData = a.getData();
        double[] bData = b.getData();

        for (int bIdx = 0; bIdx < batch; bIdx++) {
            int aOffset = bIdx * m * k;
            int bOffset = bIdx * k * n;
            int cOffset = bIdx * m * n;

            if (a.isRequiresGrad()) {
                double[] dA = a.getGrad();
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < k; j++) {
                        double sum = 0.0;
                        for (int p = 0; p < n; p++) {
                            sum += dC[cOffset + i * n + p] * bData[bOffset + j * n + p];
                        }
                        dA[aOffset + i * k + j] += sum;
                    }
                }
            }

            if (b.isRequiresGrad()) {
                double[] dB = b.getGrad();
                for (int i = 0; i < k; i++) {
                    for (int j = 0; j < n; j++) {
                        double sum = 0.0;
                        for (int p = 0; p < m; p++) {
                            sum += aData[aOffset + p * k + i] * dC[cOffset + p * n + j];
                        }
                        dB[bOffset + i * n + j] += sum;
                    }
                }
            }
        }
    }

    private static String shapeStr(Tensor t) {
        StringBuilder sb = new StringBuilder("(");
        for (int i = 0; i < t.dims(); i++) {
            if (i > 0) sb.append(",");
            sb.append(t.size(i));
        }
        return sb.append(")").toString();
    }
}
