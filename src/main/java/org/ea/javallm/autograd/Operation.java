package org.ea.javallm.autograd;

/**
 * Abstract base class for all differentiable operations in the computation graph.
 *
 * Each concrete Operation (MatMul, Add, Softmax, etc.) extends this class, implements
 * the forward computation in its own static or instance method, and implements backward()
 * to propagate gradients from the output tensor back to its inputs.
 *
 * Operations link into the graph automatically: when an Operation produces an output
 * Tensor, it sets output.creator = this, enabling the backward pass to traverse
 * the graph via creator links.
 */
public abstract class Operation {

    protected final Tensor[] inputs;
    protected final Tensor output;

    /**
     * @param inputs the input tensors consumed by this operation
     * @param output the tensor produced by this operation
     */
    protected Operation(Tensor[] inputs, Tensor output) {
        this.inputs = inputs;
        this.output = output;
        // Link the output tensor back to this operation so backward() can traverse the graph
        output.setCreator(this);
    }

    /**
     * Propagates gradients from output.grad back to each input tensor's grad array.
     *
     * Implementations MUST accumulate (+=) into input.grad rather than overwriting,
     * because a tensor may be used as input to multiple operations.
     */
    public abstract void backward();

    public Tensor[] getInputs() {
        return inputs;
    }

    public Tensor getOutput() {
        return output;
    }
}
