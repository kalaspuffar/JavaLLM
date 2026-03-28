package org.ea.javallm.autograd;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * N-dimensional tensor with flat row-major storage and automatic differentiation support.
 *
 * Data is stored in a single {@code double[]} array with precomputed strides for
 * converting multi-dimensional indices to flat offsets. This avoids the overhead of
 * nested arrays and provides cache-friendly sequential access.
 *
 * When {@code requiresGrad} is true, the tensor tracks a gradient array and a link
 * to the {@link Operation} that produced it ({@code creator}). Calling {@link #backward()}
 * on a scalar tensor performs reverse-mode automatic differentiation by traversing
 * the computation graph via creator links.
 */
public class Tensor {

    private final double[] data;
    private final int[] shape;
    private final int[] strides;
    private final boolean requiresGrad;
    private double[] grad;
    private Operation creator;
    private String name;

    /**
     * Constructs a tensor from raw data with the given shape.
     *
     * @param data         flat array of values in row-major order
     * @param shape        dimensions of the tensor (e.g., {2, 3} for a 2x3 matrix)
     * @param requiresGrad whether this tensor should track gradients
     * @throws IllegalArgumentException if data length does not match the product of shape dimensions
     */
    public Tensor(double[] data, int[] shape, boolean requiresGrad) {
        int expectedSize = computeSize(shape);
        if (data.length != expectedSize) {
            throw new IllegalArgumentException(
                    "Data length " + data.length + " does not match shape " +
                    Arrays.toString(shape) + " (expected " + expectedSize + " elements)");
        }
        this.data = data;
        this.shape = shape.clone();
        this.strides = computeStrides(shape);
        this.requiresGrad = requiresGrad;
        this.grad = requiresGrad ? new double[expectedSize] : null;
    }

    // --- Factory methods ---

    /**
     * Creates a zero-filled tensor with the given shape. Does not track gradients.
     */
    public static Tensor zeros(int... shape) {
        int size = computeSize(shape);
        return new Tensor(new double[size], shape, false);
    }

    /**
     * Creates a tensor filled with Gaussian random values scaled by {@code scale},
     * suitable for Xavier/Glorot weight initialization.
     *
     * @param shape the dimensions of the tensor
     * @param rng   random number generator (seed it for reproducibility)
     * @param scale multiplier for each sample (e.g., sqrt(2/fan_in) for He init)
     */
    public static Tensor randn(int[] shape, Random rng, double scale) {
        int size = computeSize(shape);
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = rng.nextGaussian() * scale;
        }
        return new Tensor(data, shape, false);
    }

    // --- Index computation ---

    /**
     * Converts multi-dimensional indices to a flat array offset using precomputed strides.
     *
     * For a tensor with shape {2, 3, 4} and strides {12, 4, 1}:
     * index(1, 2, 3) = 1*12 + 2*4 + 3*1 = 23
     */
    public int index(int... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException(
                    "Expected " + shape.length + " indices but got " + indices.length);
        }
        int offset = 0;
        for (int i = 0; i < indices.length; i++) {
            offset += indices[i] * strides[i];
        }
        return offset;
    }

    // --- Element access ---

    public double get(int... indices) {
        return data[index(indices)];
    }

    public void set(double value, int... indices) {
        data[index(indices)] = value;
    }

    // --- Shape queries ---

    /**
     * Returns the total number of elements in this tensor.
     */
    public int size() {
        return data.length;
    }

    /**
     * Returns the size of a specific dimension.
     */
    public int size(int dim) {
        return shape[dim];
    }

    /**
     * Returns the number of dimensions (rank) of this tensor.
     */
    public int dims() {
        return shape.length;
    }

    // --- Gradient operations ---

    /**
     * Resets all gradient values to zero. No-op if this tensor does not track gradients.
     */
    public void zeroGrad() {
        if (requiresGrad && grad != null) {
            Arrays.fill(grad, 0.0);
        }
    }

    /**
     * Performs reverse-mode automatic differentiation starting from this tensor.
     *
     * This method may only be called on a scalar (single-element) tensor. It seeds the
     * gradient to 1.0, builds a topological ordering of the computation graph using an
     * iterative depth-first traversal (no recursion, to avoid stack overflow on deep
     * graphs), and then calls each Operation's backward() in reverse order.
     *
     * @throws IllegalStateException if this tensor has more than one element
     */
    public void backward() {
        if (size() != 1) {
            throw new IllegalStateException(
                    "backward() can only be called on a scalar tensor (size=1), but this tensor has size " + size());
        }

        // Seed the gradient of the loss scalar
        if (grad == null) {
            grad = new double[1];
        }
        grad[0] = 1.0;

        // Build topological order via iterative DFS
        List<Tensor> topologicalOrder = buildTopologicalOrder();

        // Propagate gradients in reverse topological order
        for (int i = topologicalOrder.size() - 1; i >= 0; i--) {
            Tensor node = topologicalOrder.get(i);
            if (node.creator != null) {
                node.creator.backward();
            }
        }
    }

    /**
     * Iterative post-order DFS to build topological order of the computation graph.
     * Returns nodes with dependencies before dependents (leaves first, root last).
     *
     * Uses a two-pass approach: push each node twice — the first time to expand its
     * children, the second time (recognized via the visited set) to add it to the order.
     * This mirrors recursive DFS post-order without risking stack overflow.
     */
    private List<Tensor> buildTopologicalOrder() {
        List<Tensor> order = new ArrayList<>();
        Set<Tensor> visited = new HashSet<>();
        Set<Tensor> expanded = new HashSet<>();
        Deque<Tensor> stack = new ArrayDeque<>();

        stack.push(this);

        while (!stack.isEmpty()) {
            Tensor current = stack.peek();

            if (expanded.contains(current)) {
                // Second visit: all children processed, add to topological order
                stack.pop();
                if (visited.add(current)) {
                    order.add(current);
                }
            } else {
                // First visit: mark as expanded and push children
                expanded.add(current);
                if (current.creator != null) {
                    for (Tensor input : current.creator.getInputs()) {
                        if (!expanded.contains(input)) {
                            stack.push(input);
                        }
                    }
                }
            }
        }

        return order;
    }

    // --- Accessors ---

    public double[] getData() {
        return data;
    }

    public int[] getShape() {
        return shape.clone();
    }

    public int[] getStrides() {
        return strides.clone();
    }

    public boolean isRequiresGrad() {
        return requiresGrad;
    }

    public double[] getGrad() {
        return grad;
    }

    public Operation getCreator() {
        return creator;
    }

    public void setCreator(Operation creator) {
        this.creator = creator;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    // --- Internal helpers ---

    /**
     * Computes row-major (C-order) strides from shape.
     * For shape {2, 3, 4}: strides = {12, 4, 1}
     * Each stride[i] = product of shape[i+1..n-1]
     */
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

    private static int computeSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }
}
