package org.ea.javallm.autograd.ops;

import org.ea.javallm.autograd.Operation;
import org.ea.javallm.autograd.Tensor;

/**
 * Cross-entropy loss from raw logits and integer targets.
 *
 * Combines softmax and negative log-likelihood in a single operation for
 * numerical stability — avoids computing log(softmax) separately.
 *
 * Input: logits tensor of shape (batch, numClasses), targets tensor of shape (batch)
 *        where each target value is a class index in [0, numClasses).
 * Output: scalar loss = mean(-log(softmax(logits)[target])) over the batch.
 *
 * Backward uses the shortcut gradient:
 *   dLogits_i = (softmax_i - 1(i == target)) / batchSize
 * This is simpler and more numerically stable than composing softmax + NLL gradients.
 */
public class CrossEntropy extends Operation {

    // Cached softmax probabilities for backward pass
    private final double[] softmaxProbs;
    private final int[] targets;
    private final int batchSize;
    private final int numClasses;

    public CrossEntropy(Tensor logits, Tensor targetsTensor, Tensor output,
                        double[] softmaxProbs, int[] targets, int batchSize, int numClasses) {
        super(new Tensor[]{logits, targetsTensor}, output);
        this.softmaxProbs = softmaxProbs;
        this.targets = targets;
        this.batchSize = batchSize;
        this.numClasses = numClasses;
    }

    public static Tensor forward(Tensor logits, Tensor targetsTensor) {
        int batchSize = logits.size(0);
        int numClasses = logits.size(1);
        double[] logitData = logits.getData();
        double[] targetData = targetsTensor.getData();

        int[] targets = new int[batchSize];
        for (int i = 0; i < batchSize; i++) {
            targets[i] = (int) targetData[i];
        }

        double[] softmaxProbs = new double[batchSize * numClasses];
        double totalLoss = 0.0;

        for (int b = 0; b < batchSize; b++) {
            int offset = b * numClasses;

            // Numerically stable softmax: subtract max before exp
            double max = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < numClasses; c++) {
                if (logitData[offset + c] > max) {
                    max = logitData[offset + c];
                }
            }

            double sumExp = 0.0;
            for (int c = 0; c < numClasses; c++) {
                softmaxProbs[offset + c] = Math.exp(logitData[offset + c] - max);
                sumExp += softmaxProbs[offset + c];
            }

            for (int c = 0; c < numClasses; c++) {
                softmaxProbs[offset + c] /= sumExp;
            }

            // Negative log-likelihood for the target class
            // Clamp probability to avoid log(0)
            double targetProb = Math.max(softmaxProbs[offset + targets[b]], 1e-12);
            totalLoss += -Math.log(targetProb);
        }

        double meanLoss = totalLoss / batchSize;
        Tensor output = new Tensor(new double[]{meanLoss}, new int[]{1}, logits.isRequiresGrad());
        new CrossEntropy(logits, targetsTensor, output, softmaxProbs, targets, batchSize, numClasses);
        return output;
    }

    @Override
    public void backward() {
        Tensor logits = inputs[0];
        if (!logits.isRequiresGrad()) return;

        double[] dLogits = logits.getGrad();
        double outputGrad = output.getGrad()[0];

        // Shortcut gradient: dLogits_i = outputGrad * (softmax_i - 1(i==target)) / batchSize
        for (int b = 0; b < batchSize; b++) {
            int offset = b * numClasses;
            for (int c = 0; c < numClasses; c++) {
                double indicator = (c == targets[b]) ? 1.0 : 0.0;
                dLogits[offset + c] += outputGrad * (softmaxProbs[offset + c] - indicator) / batchSize;
            }
        }
    }
}
