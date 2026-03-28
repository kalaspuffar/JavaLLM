# Project Specification: JavaLLM

**Version:** 1.0
**Date:** 2026-03-28
**Author:** Solution Architect (Claude)
**Source:** `REQUIREMENTS.md` v1.0
**Status:** Draft

---

## 1. Executive Summary

JavaLLM is an educational, from-scratch implementation of the Transformer architecture in pure Java. It mirrors the philosophy of the companion JavaCNN project: **build every piece by hand to understand how it works**.

The project delivers three major subsystems:

1. **Autograd Engine** --- A PyTorch-style automatic differentiation system with a `Tensor` class, operation graph, and reverse-mode gradient computation. This is itself an educational component --- the learner will understand how frameworks like PyTorch compute gradients automatically.

2. **Transformer Building Blocks** --- Embedding, positional encoding, attention, feed-forward, layer normalization, and residual connections. Each is its own class with a clear `forward()` method that composes autograd `Tensor` operations.

3. **Two Model Configurations** built from those shared blocks:
   - **Encoder-Decoder Transformer** --- the full "Attention Is All You Need" architecture, demonstrated with a character-level string reversal task.
   - **Decoder-Only Transformer** --- the GPT-style simplification, demonstrated with character-level Shakespeare text generation.

### Key Objectives

- A Java developer can read any single class and understand the concept it implements.
- Training completes in minutes to a few hours on a CPU (not days or weeks).
- Both models visibly learn: loss decreases, output quality improves.
- The code compiles with plain `javac` and runs with plain `java` --- no build tools, no external libraries.

### Success Criteria

| # | Criterion |
|---|-----------|
| 1 | Compiles with `javac -d out $(find src -name "*.java")` |
| 2 | Decoder-only model: loss decreases, generated text improves from gibberish to recognizable English |
| 3 | Encoder-decoder model: learns string reversal to >= 80% accuracy on short strings |
| 4 | Interactive generation: user types a prompt, model produces a continuation |
| 5 | Each class is self-contained and understandable without cross-referencing many other files |

---

## 2. Architecture Overview

### 2.1 High-Level Architecture

```
JavaLLM Architecture
=====================

+--------------------------------------------------+
|                  DEMO PROGRAMS                     |
|  CharGenerationTest    TranslationTest             |
+--------------------------------------------------+
          |                        |
          v                        v
+--------------------------------------------------+
|               MODEL CONFIGURATIONS                 |
|  DecoderOnlyModel          EncoderDecoderModel     |
+--------------------------------------------------+
          |                        |
          v                        v
+--------------------------------------------------+
|            TRANSFORMER BUILDING BLOCKS             |
|  TransformerBlock  MultiHeadAttention  FeedForward  |
|  Embedding  PositionalEncoding  LayerNorm  Linear  |
+--------------------------------------------------+
          |
          v
+--------------------------------------------------+
|                 AUTOGRAD ENGINE                     |
|  Tensor  Operation (abstract)                      |
|  ops: MatMul, Add, Multiply, Softmax, ReLU,       |
|       LayerNormOp, CrossEntropy, Embedding,        |
|       Transpose, Reshape, Mask, ScalarOps          |
|  GradientChecker                                   |
+--------------------------------------------------+
          ^
          |
+-------------------+  +---------------------------+
|   DATA PIPELINE   |  |   TRAINING INFRASTRUCTURE  |
|  CharTokenizer    |  |  AdamOptimizer             |
|  TextReader       |  |  TrainingLoop (in demos)   |
|  SequenceBatcher  |  |  Model save/load           |
|  ReversalTask     |  |                            |
+-------------------+  +---------------------------+
```

### 2.2 Key Architectural Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **Autograd engine with computation graph** (not manual backprop per layer) | Educational goal: understand how PyTorch works. Layers become forward-only compositions of Tensor ops. |
| 2 | **Fine-grained operations** --- each primitive (matmul, add, relu) is a separate graph node | Maximizes educational value. Model is small enough that object overhead is irrelevant. |
| 3 | **N-dimensional Tensor with flat `double[]`** (row-major / C-order) | Java-native approach. No fancy indexing --- flat array with shape metadata and stride-based offset computation. |
| 4 | **Pre-norm architecture** (LayerNorm before sublayers) | More stable training, simpler implementation. Standard in modern Transformers. Comment explains the difference from the original paper's post-norm. |
| 5 | **Weight tying** (output projection shares weights with input embedding) | Reduces parameters, standard practice, educational concept worth exposing. |
| 6 | **ReLU activation** in feed-forward layers | Simplest, matches the original paper. GELU mentioned in comments as modern alternative. |
| 7 | **Character-level tokenization only** | Simplest tokenizer possible. Keeps vocabulary small (~65-95 tokens). BPE deferred to future phase. |
| 8 | **Plain text model persistence** | Human-readable format: named tensors with shapes and values. Stable across code changes, greppable, diffable. No brittle `Serializable` coupling. |

### 2.3 Relationship to JavaCNN

JavaLLM follows JavaCNN's conventions (package naming, project structure, one-class-per-concept) but differs in one fundamental way: **gradient computation is automatic, not manual**.

| Aspect | JavaCNN | JavaLLM |
|--------|---------|---------|
| Tensor type | `DataBlock` (3D: W x H x D) | `Tensor` (N-dimensional, grad-tracked) |
| Layer contract | `forward()` + `backward()` | `forward()` only (composes Tensor ops) |
| Gradient flow | Manual chain in each layer's `backward()` | Automatic via computation graph traversal |
| Optimizer input | `List<BackPropResult>` | Iterates over all Tensors with `requiresGrad` |
| Serialization | `Serializable` on `JavaCNN` + all layers | Plain text format: named tensors with shapes + values |

This shift is intentional and educational: the learner who already understands manual backprop from JavaCNN now learns how autograd abstracts it away.

---

## 3. System Components

### 3.1 Autograd Engine (`org.ea.javallm.autograd`)

The autograd engine is the foundation of the entire project and is itself a primary educational component.

#### 3.1.1 Tensor (`Tensor.java`)

**Purpose:** The central data structure. Wraps a multi-dimensional array of doubles with optional gradient tracking and computation graph linkage.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `data` | `double[]` | Flat array holding all values in row-major order |
| `grad` | `double[]` | Gradient array, same shape as data. `null` if `requiresGrad` is false. |
| `shape` | `int[]` | Dimensions (e.g., `{batch, seqLen, embedDim}`) |
| `strides` | `int[]` | Precomputed strides for index calculation |
| `requiresGrad` | `boolean` | Whether this tensor participates in gradient computation |
| `creator` | `Operation` | The operation that produced this tensor (null for leaf tensors) |
| `name` | `String` | Optional label for debugging (e.g., "W_Q", "embedding") |

**Key Methods:**

```java
// Construction
public Tensor(double[] data, int[] shape, boolean requiresGrad)
public static Tensor zeros(int... shape)
public static Tensor randn(int... shape, Random rng, double scale)

// Element access (primarily for debugging and testing; hot paths use raw arrays)
public double get(int... indices)
public void set(double value, int... indices)

// Shape information
public int size()              // total number of elements
public int size(int dim)       // size of a specific dimension
public int dims()              // number of dimensions

// Index computation
public int index(int... indices)  // multi-dim indices -> flat offset

// Gradient operations
public void backward()         // trigger reverse-mode autodiff (only on scalar tensors)
public void zeroGrad()         // reset grad to zeros

// Autograd operations (static methods or instance methods that return new Tensors)
// Each builds a graph node linking inputs to output via an Operation
public Tensor matmul(Tensor other)
public Tensor add(Tensor other)
public Tensor multiply(Tensor other)       // element-wise
public Tensor transpose(int dim0, int dim1)
public Tensor reshape(int... newShape)
public Tensor softmax(int dim)
public Tensor relu()
public Tensor layerNorm(int dim, Tensor gamma, Tensor beta, double eps)
public Tensor crossEntropy(Tensor targets)  // returns scalar loss
public Tensor embeddingLookup(Tensor indices)
public Tensor mask(Tensor mask, double fillValue)
public Tensor scale(double scalar)
// etc.
```

**Index Computation (Row-Major / C-Order):**

For a tensor with shape `{d0, d1, d2}`:
- `strides = {d1*d2, d2, 1}`
- `index(i, j, k) = i * strides[0] + j * strides[1] + k * strides[2]`

This is standard row-major order, natural for Java's memory layout.

**Initialization Strategies:**
- **Xavier/Glorot**: `scale = sqrt(2.0 / (fan_in + fan_out))` --- used for linear projections.
- **Zeros**: Used for biases.
- **From data**: Direct construction from a `double[]` --- used for positional encoding, masks.

**Design Notes:**
- Tensors are the "nouns" of the system; Operations are the "verbs."
- Every operation on Tensors that involves a `requiresGrad` tensor returns a new Tensor whose `creator` field points to the Operation that produced it. This builds the computation graph implicitly.
- `backward()` can only be called on a scalar (single-element) tensor, typically the loss.

#### 3.1.2 Operation (`Operation.java`)

**Purpose:** Abstract base class for all differentiable operations. Each subclass knows how to propagate gradients backward through itself.

```java
public abstract class Operation {
    protected Tensor[] inputs;
    protected Tensor output;

    /**
     * Propagate gradients from output.grad back to each input's .grad.
     * Called during backward pass in reverse topological order.
     */
    public abstract void backward();
}
```

**Design Notes:**
- Each Operation stores references to its inputs and output.
- During the forward pass, the Operation is created and linked into the graph.
- During the backward pass, `backward()` reads `output.grad` and accumulates gradients into each input's `.grad` field.
- Gradients **accumulate** (+=) rather than replace, to handle cases where a tensor is used in multiple operations.

#### 3.1.3 Operations (`org.ea.javallm.autograd.ops`)

Each operation is a separate class. This is fine-grained by design --- the learner sees exactly how gradients flow through each primitive.

##### MatMul (`MatMul.java`)
- **Forward:** `C = A @ B` (2D matrix multiply, extended to handle batched 3D tensors)
- **Backward:** `dA = dC @ B^T`, `dB = A^T @ dC`
- **Shapes:** For 2D: `(m,k) @ (k,n) -> (m,n)`. For 3D (batched): `(b,m,k) @ (b,k,n) -> (b,m,n)`.
- **Implementation:** Triple nested loop for 2D; outer batch loop for 3D. No BLAS --- clarity over speed.

##### Add (`Add.java`)
- **Forward:** `C = A + B` (element-wise, with broadcasting support for bias addition)
- **Backward:** `dA = dC`, `dB = dC` (summed over broadcast dimensions if shapes differ)
- **Broadcasting:** Supports adding a `(1, n)` or `(n)` bias to a `(batch, seq, n)` tensor.

##### Multiply (`Multiply.java`)
- **Forward:** `C = A * B` (element-wise / Hadamard product)
- **Backward:** `dA = dC * B`, `dB = dC * A`

##### Scale (`Scale.java`)
- **Forward:** `C = A * scalar`
- **Backward:** `dA = dC * scalar`
- Used for the `1/sqrt(d_k)` scaling in attention.

##### Softmax (`Softmax.java`)
- **Forward:** Numerically stable softmax along a specified dimension.
  - Subtract max for stability: `exp(x_i - max(x)) / sum(exp(x_j - max(x)))`
- **Backward:** Jacobian-vector product: `dA_i = S_i * (dC_i - sum(dC_j * S_j))`
  - Where `S` is the softmax output.
- **Dimension parameter:** Specifies which axis to normalize along (typically the last axis).

##### ReLU (`ReLU.java`)
- **Forward:** `C = max(0, A)`
- **Backward:** `dA = dC * (A > 0 ? 1 : 0)`
- Stores a boolean mask or references the input for the backward pass.

##### Transpose (`Transpose.java`)
- **Forward:** Swaps two dimensions. Implemented by changing shape and strides (logical transpose); a physical copy is made to keep downstream operations simple.
- **Backward:** Transpose the gradient back using the inverse permutation.

##### Reshape (`Reshape.java`)
- **Forward:** Changes shape without altering data (same flat array, new shape/strides).
- **Backward:** Reshapes gradient back to original shape.
- **Constraint:** Total element count must be preserved.

##### LayerNormOp (`LayerNormOp.java`)
- **Forward:** Normalize across the last dimension:
  - `mean = avg(x)`, `var = avg((x - mean)^2)`
  - `x_norm = (x - mean) / sqrt(var + eps)`
  - `output = gamma * x_norm + beta`
- **Backward:** Gradients for gamma, beta, and the input. This is one of the more complex backward passes; the code should include comments showing the derivation.
- **References:** Layer Normalization (Ba et al., 2016).

##### CrossEntropyLoss (`CrossEntropy.java`)
- **Forward:** Takes logits tensor and integer target indices. Computes:
  - `loss = -log(softmax(logits)[target])`
  - Averaged over the batch.
- **Backward:** `dLogits_i = softmax(logits)_i - (1 if i == target else 0)` (the classic softmax + cross-entropy gradient shortcut).
- Returns a scalar tensor (single element).

##### EmbeddingLookup (`EmbeddingLookup.java`)
- **Forward:** Given a weight matrix of shape `(vocab_size, embed_dim)` and integer indices, select rows.
  - Output shape: indices shape + `(embed_dim,)`.
- **Backward:** Scatter gradients back to the selected rows of the weight matrix. Only the looked-up rows receive gradient.

##### Mask (`Mask.java`)
- **Forward:** `C = A` where mask is true/1; `C = fillValue` where mask is false/0.
  - `fillValue` is typically `-1e9` (large negative, becomes ~0 after softmax).
- **Backward:** `dA = dC * (mask == true ? 1 : 0)`. Masked positions receive no gradient.

#### 3.1.4 GradientChecker (`GradientChecker.java`)

**Purpose:** Numerical gradient verification utility. Compares autograd gradients against finite-difference approximations to catch bugs in Operation implementations.

**Algorithm:**
```
For each parameter p_i:
    original = p_i
    p_i = original + epsilon
    loss_plus = forward()
    p_i = original - epsilon
    loss_minus = forward()
    p_i = original  // restore
    numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
    Compare numerical_grad against autograd_grad
    Report if relative error > threshold
```

**Parameters:**
- `epsilon`: Perturbation size (default `1e-5`)
- `threshold`: Maximum acceptable relative error (default `1e-4`)

**Usage:** Called in test/demo programs to validate each operation before building larger models.

#### 3.1.5 Backward Pass Algorithm

When `loss.backward()` is called:

1. Set `loss.grad = 1.0` (seed gradient).
2. Build a topological ordering of all nodes reachable from `loss` by traversing `creator` links.
3. Process nodes in reverse topological order (from loss back to inputs).
4. For each node, call `node.creator.backward()` which accumulates gradients into input tensors.

**Implementation:** Iterative topological sort using visited set and stack. No recursion (avoids stack overflow for deep graphs).

```java
public void backward() {
    // 1. Seed gradient
    this.grad = new double[]{1.0};

    // 2. Topological sort
    List<Tensor> sorted = topologicalSort(this);

    // 3. Reverse pass
    for (int i = sorted.size() - 1; i >= 0; i--) {
        Tensor t = sorted.get(i);
        if (t.creator != null) {
            t.creator.backward();
        }
    }
}
```

### 3.2 Transformer Building Blocks (`org.ea.javallm.layers`)

Each layer class has a `forward()` method that takes Tensor inputs and returns Tensor outputs. Because they compose autograd operations, **no `backward()` method is needed** --- the computation graph handles gradient flow automatically.

All learnable parameters are stored as `Tensor` fields with `requiresGrad = true`.

#### 3.2.1 Linear (`Linear.java`)

**Purpose:** A linear projection: `output = input @ W^T + b`. The most basic building block.

**Parameters:**
| Parameter | Shape | Description |
|-----------|-------|-------------|
| `weight` | `(outputDim, inputDim)` | Weight matrix (transposed for efficient matmul) |
| `bias` | `(outputDim)` | Bias vector (optional) |

**Forward:**
```
output = input.matmul(weight.transpose(0, 1)) + bias
```
- Input shape: `(batch, seqLen, inputDim)` or `(batch, inputDim)`
- Output shape: same but with `inputDim` replaced by `outputDim`

**Initialization:** Xavier/Glorot: `scale = sqrt(2.0 / (inputDim + outputDim))`

#### 3.2.2 Embedding (`Embedding.java`)

**Purpose:** Maps integer token IDs to dense vectors. Implements the token embedding from "Attention Is All You Need" Section 3.4.

**Parameters:**
| Parameter | Shape | Description |
|-----------|-------|-------------|
| `weight` | `(vocabSize, embedDim)` | Embedding matrix |

**Forward:**
```
output = weight.embeddingLookup(tokenIndices)
```
- Input: integer array of shape `(batch, seqLen)` (token IDs)
- Output: `(batch, seqLen, embedDim)` tensor

**Note on Weight Tying:** The decoder-only model's output projection reuses this weight matrix (transposed) to predict next-token logits. This is a standard technique that reduces parameters and is documented in the code.

#### 3.2.3 PositionalEncoding (`PositionalEncoding.java`)

**Purpose:** Adds position information to embeddings using sinusoidal functions from "Attention Is All You Need" Section 3.5.

**Not learnable** --- computed once at construction based on maximum sequence length and embedding dimension.

**Formula:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Forward:**
```
output = input + positionalEncodingTable[0:seqLen]
```
- The PE table is stored as a non-gradient Tensor of shape `(maxSeqLen, embedDim)`.
- Only the first `seqLen` rows are added (sliced to match input length).

**Design Note:** The sinusoidal encoding is fixed, not learned. This is simpler and matches the original paper. Comments explain that modern models often use learned or rotary positional encodings instead.

#### 3.2.4 MultiHeadAttention (`MultiHeadAttention.java`)

**Purpose:** The core attention mechanism. Implements Section 3.2 of "Attention Is All You Need."

**Parameters:**
| Parameter | Shape | Description |
|-----------|-------|-------------|
| `W_Q` | Linear(embedDim, embedDim) | Query projection |
| `W_K` | Linear(embedDim, embedDim) | Key projection |
| `W_V` | Linear(embedDim, embedDim) | Value projection |
| `W_O` | Linear(embedDim, embedDim) | Output projection |

**Configuration:**
| Config | Description |
|--------|-------------|
| `numHeads` | Number of attention heads |
| `headDim` | `embedDim / numHeads` |

**Forward (Self-Attention):**
```
Q = W_Q.forward(x)                        // (batch, seqLen, embedDim)
K = W_K.forward(x)
V = W_V.forward(x)

// Split into heads: (batch, seqLen, embedDim) -> (batch, numHeads, seqLen, headDim)
Q = Q.reshape(batch, seqLen, numHeads, headDim).transpose(1, 2)
K = K.reshape(batch, seqLen, numHeads, headDim).transpose(1, 2)
V = V.reshape(batch, seqLen, numHeads, headDim).transpose(1, 2)

// Scaled dot-product attention
scores = Q.matmul(K.transpose(2, 3)).scale(1.0 / sqrt(headDim))
if (mask != null) scores = scores.mask(causalMask, -1e9)
weights = scores.softmax(dim=3)
attended = weights.matmul(V)               // (batch, numHeads, seqLen, headDim)

// Concatenate heads: -> (batch, seqLen, embedDim)
attended = attended.transpose(1, 2).reshape(batch, seqLen, embedDim)

output = W_O.forward(attended)
```

**Forward (Cross-Attention):**
Same as above, but Q comes from the decoder input while K and V come from the encoder output. The method signature accepts separate `query`, `keyValue` tensors.

```java
public Tensor forward(Tensor query, Tensor keyValue, Tensor mask)
```
For self-attention, `query == keyValue`. For cross-attention, they differ.

**Causal Mask:** An upper-triangular matrix of shape `(seqLen, seqLen)` where positions that should be masked are 0 and allowed positions are 1. Created once and reused.

#### 3.2.5 FeedForward (`FeedForward.java`)

**Purpose:** Position-wise feed-forward network. Implements Section 3.3 of "Attention Is All You Need."

**Parameters:**
| Parameter | Component | Description |
|-----------|-----------|-------------|
| `linear1` | `Linear(embedDim, innerDim)` | First projection (expand) |
| `linear2` | `Linear(innerDim, embedDim)` | Second projection (contract) |

**Forward:**
```
output = linear2.forward(linear1.forward(x).relu())
```

**Configuration:** `innerDim` is typically `4 * embedDim`.

**Design Note:** ReLU is used following the original paper. A comment notes that GPT-2 and later models use GELU instead.

#### 3.2.6 LayerNorm (`LayerNorm.java`)

**Purpose:** Layer normalization across the feature dimension. Stabilizes training by normalizing activations.

**Parameters:**
| Parameter | Shape | Description |
|-----------|-------|-------------|
| `gamma` | `(embedDim)` | Learnable scale, initialized to 1.0 |
| `beta` | `(embedDim)` | Learnable shift, initialized to 0.0 |

**Forward:**
```
output = x.layerNorm(dim=-1, gamma, beta, eps=1e-5)
```

This delegates to the `LayerNormOp` autograd operation which handles the math.

### 3.3 Model Configurations (`org.ea.javallm.model`)

#### 3.3.1 TransformerBlock (`TransformerBlock.java`)

**Purpose:** A single Transformer layer. Composed differently depending on whether it's used as an encoder block, a decoder block, or a decoder block with cross-attention.

**Components:**
| Component | Type | Description |
|-----------|------|-------------|
| `selfAttnNorm` | `LayerNorm` | Pre-norm before self-attention |
| `selfAttn` | `MultiHeadAttention` | Self-attention sublayer |
| `crossAttnNorm` | `LayerNorm` | Pre-norm before cross-attention (decoder with cross-attn only) |
| `crossAttn` | `MultiHeadAttention` | Cross-attention sublayer (decoder with cross-attn only) |
| `ffnNorm` | `LayerNorm` | Pre-norm before feed-forward |
| `ffn` | `FeedForward` | Feed-forward sublayer |

**Configuration:**
| Config | Description |
|--------|-------------|
| `hasCausalMask` | Whether self-attention uses causal masking (true for decoder blocks) |
| `hasCrossAttention` | Whether this block includes cross-attention (true for encoder-decoder's decoder) |

**Forward (pre-norm with residual connections):**
```
// Self-attention with residual
x = x + selfAttn.forward(selfAttnNorm.forward(x), selfAttnNorm.forward(x), causalMask)

// Cross-attention with residual (if applicable)
if (hasCrossAttention):
    x = x + crossAttn.forward(crossAttnNorm.forward(x), encoderOutput, null)

// Feed-forward with residual
x = x + ffn.forward(ffnNorm.forward(x))

return x
```

**Design Note:** This uses **pre-norm** (LayerNorm before each sublayer), which is more stable than the original paper's post-norm (LayerNorm after residual add). A comment in the code explains the difference and why modern models prefer pre-norm.

#### 3.3.2 DecoderOnlyModel (`DecoderOnlyModel.java`)

**Purpose:** GPT-style decoder-only Transformer for text generation. This is the simpler of the two model configurations and the primary focus for the Shakespeare demo.

**Components:**
| Component | Type | Description |
|-----------|------|-------------|
| `embedding` | `Embedding` | Token embedding (shared with output projection via weight tying) |
| `positionalEncoding` | `PositionalEncoding` | Sinusoidal position encoding |
| `blocks` | `List<TransformerBlock>` | Stack of N decoder blocks (causal mask, no cross-attention) |
| `finalNorm` | `LayerNorm` | Final layer norm before output projection |

**Configuration:**
| Config | Default | Description |
|--------|---------|-------------|
| `vocabSize` | from tokenizer | Number of unique tokens |
| `embedDim` | 128 | Embedding/model dimension |
| `numLayers` | 4 | Number of Transformer blocks |
| `numHeads` | 4 | Attention heads per block |
| `ffnInnerDim` | 512 | Feed-forward inner dimension |
| `maxSeqLen` | 128 | Maximum context window |

**Forward:**
```java
public Tensor forward(int[] tokenIds) {
    Tensor x = embedding.forward(tokenIds);       // (batch, seqLen, embedDim)
    x = positionalEncoding.forward(x);            // add position info

    for (TransformerBlock block : blocks) {
        x = block.forward(x, null);               // null = no encoder output
    }

    x = finalNorm.forward(x);

    // Output projection: reuse embedding weights (weight tying)
    // logits = x @ embedding.weight^T
    Tensor logits = x.matmul(embedding.weight.transpose(0, 1));

    return logits;   // (batch, seqLen, vocabSize)
}
```

**Weight Tying:** The output projection multiplies by the embedding weight matrix (transposed). This means the same matrix that converts token IDs to vectors also converts vectors back to token probabilities. It reduces parameters and provides a useful inductive bias. A comment explains why this works.

**Causal Mask:** Created once as a lower-triangular matrix of shape `(maxSeqLen, maxSeqLen)`. Positions `(i, j)` where `j > i` are masked (set to 0), preventing each position from attending to future positions.

**Parameter Collection:** A `getParameters()` method returns all `Tensor` objects with `requiresGrad = true` across all sub-components. The optimizer iterates over this list.

```java
public List<Tensor> getParameters() {
    List<Tensor> params = new ArrayList<>();
    params.add(embedding.weight);
    for (TransformerBlock block : blocks) {
        params.addAll(block.getParameters());
    }
    params.addAll(finalNorm.getParameters());
    // Note: output projection weight is embedding.weight (tied), not added again
    return params;
}
```

#### 3.3.3 EncoderDecoderModel (`EncoderDecoderModel.java`)

**Purpose:** Full encoder-decoder Transformer from "Attention Is All You Need." Demonstrated with the string reversal task.

**Components:**
| Component | Type | Description |
|-----------|------|-------------|
| `srcEmbedding` | `Embedding` | Source (encoder) token embedding |
| `tgtEmbedding` | `Embedding` | Target (decoder) token embedding |
| `positionalEncoding` | `PositionalEncoding` | Shared positional encoding (same formula, reused) |
| `encoderBlocks` | `List<TransformerBlock>` | Stack of N encoder blocks (no causal mask, no cross-attn) |
| `decoderBlocks` | `List<TransformerBlock>` | Stack of N decoder blocks (causal mask + cross-attention) |
| `finalNorm` | `LayerNorm` | Final layer norm |
| `outputProjection` | `Linear` | Linear projection to vocabulary size |

**Configuration:**
| Config | Default | Description |
|--------|---------|-------------|
| `srcVocabSize` | from tokenizer | Source vocabulary size |
| `tgtVocabSize` | from tokenizer | Target vocabulary size |
| `embedDim` | 64 | Embedding dimension |
| `numLayers` | 2 | Blocks per encoder and decoder |
| `numHeads` | 2 | Attention heads |
| `ffnInnerDim` | 256 | Feed-forward inner dimension |
| `maxSeqLen` | 32 | Maximum sequence length |

**Forward:**
```java
public Tensor forward(int[] srcTokenIds, int[] tgtTokenIds) {
    // Encode
    Tensor enc = srcEmbedding.forward(srcTokenIds);
    enc = positionalEncoding.forward(enc);
    for (TransformerBlock block : encoderBlocks) {
        enc = block.forward(enc, null);  // self-attention only, no mask
    }

    // Decode
    Tensor dec = tgtEmbedding.forward(tgtTokenIds);
    dec = positionalEncoding.forward(dec);
    for (TransformerBlock block : decoderBlocks) {
        dec = block.forward(dec, enc);   // self-attention + cross-attention to encoder
    }

    dec = finalNorm.forward(dec);
    Tensor logits = outputProjection.forward(dec);  // (batch, seqLen, tgtVocabSize)
    return logits;
}
```

**Design Note:** The encoder-decoder model uses a **separate** output Linear projection (no weight tying) because source and target vocabularies may differ. For the string reversal task they share the same character set, but the architecture supports different vocabularies. A comment explains the contrast with the decoder-only model's weight tying.

### 3.4 Data Pipeline (`org.ea.javallm.data`)

#### 3.4.1 CharTokenizer (`CharTokenizer.java`)

**Purpose:** Character-level tokenizer. Maps individual characters to integer IDs and back.

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `charToIndex` | `Map<Character, Integer>` | Character to ID mapping |
| `indexToChar` | `char[]` | ID to character mapping |
| `vocabSize` | `int` | Number of unique tokens |

**Special Tokens (for encoder-decoder):**
| Token | ID | Purpose |
|-------|----|---------|
| `<PAD>` | 0 | Padding for variable-length sequences |
| `<SOS>` | 1 | Start of sequence (decoder input prefix) |
| `<EOS>` | 2 | End of sequence (signals completion) |

For the decoder-only model, special tokens are not strictly needed (the vocabulary is just the corpus characters), but the tokenizer supports them.

**Methods:**
```java
public static CharTokenizer fromText(String text)       // scan text, build vocab
public static CharTokenizer fromText(String text, boolean includeSpecialTokens)
public int[] encode(String text)                         // string -> int[]
public String decode(int[] tokens)                       // int[] -> string
public int getVocabSize()
```

#### 3.4.2 TextReader (`TextReader.java`)

**Purpose:** Reads a plain text file into memory and splits into train/validation portions.

**Methods:**
```java
public TextReader(String filePath, double trainSplit)    // default: 0.9
public String getTrainText()
public String getValidationText()
```

Simple and straightforward --- reads the entire file into a `String`, splits at the 90% mark.

#### 3.4.3 SequenceBatcher (`SequenceBatcher.java`)

**Purpose:** Produces fixed-length training batches from tokenized text for the decoder-only model.

**Behavior:**
- Takes tokenized text (full `int[]` of token IDs) and a context window size.
- Produces batches of `(input, target)` pairs where:
  - `input  = tokens[pos : pos + contextLen]`
  - `target = tokens[pos + 1 : pos + contextLen + 1]` (shifted by one)
- Batches are created by sampling random starting positions.
- Shuffles between epochs.

**Methods:**
```java
public SequenceBatcher(int[] tokens, int contextLen, int batchSize, Random rng)
public boolean hasNext()
public int[][] nextInputBatch()     // (batchSize, contextLen)
public int[][] nextTargetBatch()    // (batchSize, contextLen)
public void reset()                 // reshuffle for next epoch
```

#### 3.4.4 ReversalTaskGenerator (`ReversalTaskGenerator.java`)

**Purpose:** Generates synthetic string reversal pairs for the encoder-decoder demo.

**Behavior:**
- Generates random strings of characters (configurable alphabet and length range).
- Each pair: input = original string, output = reversed string.
- Both are tokenized using the shared CharTokenizer.
- Target sequences are prefixed with `<SOS>` and suffixed with `<EOS>`.

**Methods:**
```java
public ReversalTaskGenerator(CharTokenizer tokenizer, int minLen, int maxLen, Random rng)
public void generateBatch(int batchSize,
                          int[][] srcOut, int[][] tgtInputOut, int[][] tgtTargetOut)
```
- `srcOut`: encoder input (original string, padded)
- `tgtInputOut`: decoder input (SOS + reversed string, for teacher forcing)
- `tgtTargetOut`: decoder target (reversed string + EOS, what we want it to predict)

### 3.5 Training Infrastructure (`org.ea.javallm.trainers`)

#### 3.5.1 AdamOptimizer (`AdamOptimizer.java`)

**Purpose:** The Adam optimizer (Kingma & Ba, 2014). Standard for Transformer training.

**Algorithm:**
```
For each parameter tensor p with gradient g:
    t += 1
    m = beta1 * m + (1 - beta1) * g          // first moment estimate
    v = beta2 * v + (1 - beta2) * g^2        // second moment estimate
    m_hat = m / (1 - beta1^t)                // bias correction
    v_hat = v / (1 - beta2^t)                // bias correction
    p -= learningRate * m_hat / (sqrt(v_hat) + eps)
```

**Fields:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `learningRate` | `double` | 3e-4 | Step size |
| `beta1` | `double` | 0.9 | First moment decay |
| `beta2` | `double` | 0.999 | Second moment decay |
| `epsilon` | `double` | 1e-8 | Numerical stability |
| `t` | `int` | 0 | Timestep counter |
| `m` | `Map<Tensor, double[]>` | --- | First moment per parameter |
| `v` | `Map<Tensor, double[]>` | --- | Second moment per parameter |

**Methods:**
```java
public AdamOptimizer(List<Tensor> parameters, double learningRate)
public void step()       // apply one update to all parameters
public void zeroGrad()   // zero all parameter gradients
```

**Design Note:** Unlike JavaCNN where the trainer calls `forward/backward/update` as a monolithic operation, here the optimizer is decoupled. The training loop calls `forward`, `loss.backward()`, `optimizer.step()`, `optimizer.zeroGrad()` as separate steps. This mirrors PyTorch's API and is more educational.

#### 3.5.2 ModelIO (`ModelIO.java`)

**Purpose:** Save and load model parameters in a human-readable plain text format. Each parameter tensor is written as a named section with its shape and values.

**Why not Java Serialization:** Java's `ObjectOutputStream` couples the binary format to the exact class structure. Renaming a field, moving a class, or adding a method breaks deserialization. During an educational project where code is constantly being tweaked, this is unacceptable. A plain text format is stable, readable, and debuggable.

**File Format:**
```
--- embedding.weight [65 128]
 0.012345 -0.045678  0.078901 -0.001234  0.056789
 0.034567 -0.089012  0.012345  0.067890 -0.023456
...

--- block.0.selfAttnNorm.gamma [128]
 1.000000  1.000000  1.000000  1.000000  1.000000
...

--- block.0.selfAttn.W_Q.weight [128 128]
 0.023456 -0.012345  0.045678 ...
...
```

**Format Rules:**
- Each tensor starts with a header line: `--- <name> [<dim0> <dim1> ...]`
- Values follow as space-separated doubles, any number per line
- Blank lines between tensors are ignored
- Lines starting with `#` are comments (for metadata like training step, date, hyperparameters)

**Methods:**
```java
public class ModelIO {
    /**
     * Save all named parameters to a text file.
     * Parameters are obtained from model.getNamedParameters() which returns
     * a map of String name -> Tensor.
     */
    public static void save(Map<String, Tensor> namedParameters, String filePath)

    /**
     * Load parameter values from a text file into existing tensors.
     * Matches by name. Throws if shapes don't match.
     */
    public static void load(Map<String, Tensor> namedParameters, String filePath)
}
```

**Named Parameters:** Each model class provides a `getNamedParameters()` method that returns a `Map<String, Tensor>` with hierarchical names (e.g., `block.0.selfAttn.W_Q.weight`). This naming convention makes model files self-documenting --- you can open the file and see exactly which component each tensor belongs to.

**Design Note:** This teaches an important concept: a trained model is nothing more than a bag of named tensors. The architecture (how those tensors are connected) lives in the code. The checkpoint (what values those tensors hold) lives in the file. This is how real formats like SafeTensors and GGUF work conceptually.

### 3.6 Demo Programs

#### 3.6.1 CharGenerationTest (`CharGenerationTest.java`)

**Purpose:** End-to-end decoder-only Transformer demo. Trains on Shakespeare, generates text.

**Program Flow:**

```
1. Load Data
   - Read shakespeare.txt via TextReader
   - Build CharTokenizer from training text
   - Create SequenceBatcher

2. Build Model
   - Create DecoderOnlyModel with hyperparameters:
     embedDim=128, numLayers=4, numHeads=4, ffnInnerDim=512, maxSeqLen=128
   - Create AdamOptimizer with lr=3e-4

3. Training Loop (configurable epochs/steps)
   For each step:
     - Get next batch from SequenceBatcher
     - Forward pass: logits = model.forward(inputBatch)
     - Compute loss: loss = logits.crossEntropy(targetBatch)
     - Backward pass: loss.backward()
     - Optimizer step: optimizer.step(), optimizer.zeroGrad()
     - Every N steps: print loss
     - Every M steps: generate and print a sample text

4. Save Model (optional)
   - Save model parameters to plain text file

5. Interactive Generation
   - Loop: read prompt from stdin
   - Generate continuation using autoregressive decoding
   - Print result
   - Repeat until user quits
```

**Sample Output During Training:**
```
Step 100 | Loss: 3.42 | tokens/sec: 1234
Step 200 | Loss: 2.87 | tokens/sec: 1256
--- Sample (step 200, temperature=0.8) ---
ROMEO: Thae sthe wou  nd hee art
--- End sample ---
Step 500 | Loss: 1.95 | tokens/sec: 1189
--- Sample (step 500, temperature=0.8) ---
ROMEO: What is the matter that you have such a face?
--- End sample ---
```

**Generation Algorithm (Autoregressive):**
```java
public String generate(DecoderOnlyModel model, CharTokenizer tokenizer,
                       String prompt, int maxLen, double temperature, Random rng) {
    int[] tokens = tokenizer.encode(prompt);

    for (int i = 0; i < maxLen; i++) {
        // Use last contextLen tokens as input (sliding window)
        int[] context = lastN(tokens, model.maxSeqLen);

        // Forward pass (no grad tracking needed for inference)
        Tensor logits = model.forward(context);

        // Take logits for the last position
        double[] lastLogits = logits.getLastPosition();

        // Apply temperature
        for (int j = 0; j < lastLogits.length; j++)
            lastLogits[j] /= temperature;

        // Sample from distribution
        double[] probs = softmax(lastLogits);
        int nextToken = sample(probs, rng);

        tokens = append(tokens, nextToken);
    }

    return tokenizer.decode(tokens);
}
```

#### 3.6.2 TranslationTest (`TranslationTest.java`)

**Purpose:** End-to-end encoder-decoder Transformer demo. Trains on string reversal.

**Program Flow:**

```
1. Setup
   - Create CharTokenizer from alphabet (a-z, plus special tokens)
   - Create ReversalTaskGenerator
   - Build EncoderDecoderModel:
     embedDim=64, numLayers=2, numHeads=2, ffnInnerDim=256, maxSeqLen=32
   - Create AdamOptimizer with lr=1e-3

2. Training Loop
   For each step:
     - Generate a batch of reversal pairs
     - Forward pass: logits = model.forward(srcBatch, tgtInputBatch)
     - Compute loss: loss = logits.crossEntropy(tgtTargetBatch)
     - Backward/optimize
     - Every N steps: print loss and accuracy
     - Every M steps: print example predictions vs expected

3. Evaluation
   - Generate test examples not seen during training
   - For each: encode, decode autoregressively, compare to expected
   - Print accuracy and examples

4. Interactive Mode (optional)
   - Read string from stdin
   - Reverse it using the model
   - Print model output vs correct answer
```

**Autoregressive Decoding for Encoder-Decoder:**
```java
public String decode(EncoderDecoderModel model, int[] srcTokens,
                     CharTokenizer tokenizer, int maxLen) {
    // Encode source once
    // Then generate target tokens one by one:
    int[] generated = {SOS_TOKEN};

    for (int i = 0; i < maxLen; i++) {
        Tensor logits = model.forward(srcTokens, generated);
        int nextToken = argmax(logits.getLastPosition());  // greedy for evaluation
        if (nextToken == EOS_TOKEN) break;
        generated = append(generated, nextToken);
    }

    return tokenizer.decode(generated);  // skip SOS
}
```

---

## 4. Data Architecture

### 4.1 Data Models

#### Tensor Internal Layout

All data flows through the `Tensor` class. There are no separate data model entities --- this is a neural network, not a database application.

**Row-major storage example:**

A tensor with shape `{2, 3, 4}` (2 batches, 3 rows, 4 columns) is stored as:
```
Flat index:  [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]  [8]  [9] [10] [11] ...
Logical:   [0,0,0][0,0,1][0,0,2][0,0,3][0,1,0][0,1,1][0,1,2][0,1,3][0,2,0]...
Strides:     12     4      1
```

`index(b, r, c) = b * 12 + r * 4 + c * 1`

#### Character Vocabulary

Built dynamically from the training corpus:
```
Shakespeare corpus -> unique chars -> sorted -> assigned sequential IDs

Example:
  ' '  -> 0
  '!'  -> 1
  ','  -> 2
  ...
  'A'  -> 10
  'B'  -> 11
  ...
  'a'  -> 36
  'b'  -> 37
  ...
  'z'  -> 61
```

For encoder-decoder tasks, special tokens (PAD=0, SOS=1, EOS=2) are inserted first, shifting character IDs up by 3.

### 4.2 File Formats

| File | Format | Description |
|------|--------|-------------|
| `data/shakespeare.txt` | UTF-8 text | Training corpus (~100KB-1MB) |
| `*.model` (saved models) | Plain text (named tensors) | One section per parameter: name, shape, values (see Section 3.5.2) |

---

## 5. API Specifications

This project has no external API. All interfaces are internal Java method calls. The key public interfaces are documented in Section 3 (component specifications).

### 5.1 Primary User-Facing Interface: Command Line

**CharGenerationTest:**
```bash
# Compile
javac -d out $(find src -name "*.java")

# Run with defaults
java -cp out org.ea.javallm.CharGenerationTest

# The program trains, then enters interactive mode:
# > To be or not to be
# (model generates continuation)
# > Exit
```

**TranslationTest:**
```bash
java -cp out org.ea.javallm.TranslationTest

# The program trains, evaluates, then optionally enters interactive mode
```

### 5.2 Internal API Summary

**Model forward pass:**
```java
// Decoder-only
Tensor logits = decoderOnlyModel.forward(int[][] inputTokens);

// Encoder-decoder
Tensor logits = encoderDecoderModel.forward(int[][] srcTokens, int[][] tgtTokens);
```

**Loss computation:**
```java
Tensor loss = logits.crossEntropy(targetTokens);
```

**Training step:**
```java
loss.backward();
optimizer.step();
optimizer.zeroGrad();
```

**Text generation:**
```java
String output = Generator.generate(model, tokenizer, prompt, maxLen, temperature, rng);
```

---

## 6. Security Architecture

Not applicable. This is a local, single-user, educational project with no network access, no authentication, and no sensitive data. The Shakespeare corpus is public domain.

---

## 7. Infrastructure and Deployment

### 7.1 Requirements

| Requirement | Detail |
|-------------|--------|
| **JDK** | 17+ (JDK 21 available on target system) |
| **Memory** | Default JVM heap is sufficient (256MB-1GB). The model has ~300K parameters at most. |
| **Disk** | < 10MB total (source + training data + saved model) |
| **OS** | Any OS with a JDK (developed on Linux) |

### 7.2 Build and Run

```bash
cd JavaLLM

# Compile all source files
javac -d out $(find src -name "*.java")

# Run decoder-only demo (Shakespeare text generation)
java -cp out org.ea.javallm.CharGenerationTest

# Run encoder-decoder demo (string reversal)
java -cp out org.ea.javallm.TranslationTest
```

No build tools (Maven, Gradle) are required. No external JARs.

### 7.3 Project Directory Structure

```
JavaLLM/
├── REQUIREMENTS.md
├── SPECIFICATION.md
├── data/
│   └── shakespeare.txt                    # Training corpus (Tiny Shakespeare, public domain)
├── src/org/ea/javallm/
│   ├── autograd/
│   │   ├── Tensor.java                    # Core tensor with gradient tracking
│   │   ├── Operation.java                 # Abstract base for differentiable operations
│   │   ├── GradientChecker.java           # Numerical gradient verification
│   │   └── ops/
│   │       ├── MatMul.java                # Matrix multiplication
│   │       ├── Add.java                   # Element-wise addition (with broadcasting)
│   │       ├── Multiply.java              # Element-wise multiplication
│   │       ├── Scale.java                 # Scalar multiplication
│   │       ├── Softmax.java               # Numerically stable softmax
│   │       ├── ReLU.java                  # ReLU activation
│   │       ├── Transpose.java             # Dimension transposition
│   │       ├── Reshape.java               # Shape change (same data)
│   │       ├── LayerNormOp.java           # Layer normalization math
│   │       ├── CrossEntropy.java          # Cross-entropy loss
│   │       ├── EmbeddingLookup.java       # Row selection by index
│   │       └── Mask.java                  # Apply mask with fill value
│   ├── layers/
│   │   ├── Linear.java                    # Linear projection (Wx + b)
│   │   ├── Embedding.java                 # Token embedding lookup
│   │   ├── PositionalEncoding.java        # Sinusoidal positional encoding
│   │   ├── MultiHeadAttention.java        # Multi-head attention mechanism
│   │   ├── FeedForward.java               # Position-wise feed-forward network
│   │   └── LayerNorm.java                 # Layer normalization
│   ├── model/
│   │   ├── TransformerBlock.java          # Single transformer block (configurable)
│   │   ├── DecoderOnlyModel.java          # GPT-style decoder-only transformer
│   │   └── EncoderDecoderModel.java       # Full encoder-decoder transformer
│   ├── data/
│   │   ├── CharTokenizer.java             # Character-level tokenizer
│   │   ├── TextReader.java                # Text file reader with train/val split
│   │   ├── SequenceBatcher.java           # Batch producer for decoder-only training
│   │   └── ReversalTaskGenerator.java     # Synthetic data for encoder-decoder demo
│   ├── trainers/
│   │   ├── AdamOptimizer.java             # Adam optimizer
│   │   └── ModelIO.java                   # Plain text model save/load
│   ├── CharGenerationTest.java            # Decoder-only demo (Shakespeare)
│   └── TranslationTest.java              # Encoder-decoder demo (string reversal)
```

---

## 8. Integration Points

None. JavaLLM is fully self-contained. It shares a repository with JavaCNN but has no code dependencies on it.

---

## 9. Testing Strategy

### 9.1 Gradient Checking (Primary Correctness Validation)

The `GradientChecker` is the most critical testing tool. Each autograd operation must pass numerical gradient verification.

**Approach:**
- For each Operation class, create a small test case with random inputs.
- Run forward + backward via autograd.
- Run forward with finite differences (numerical gradient).
- Compare. Relative error should be < 1e-4.

**What to check:**
| Operation | Test Input Shape | Notes |
|-----------|-----------------|-------|
| MatMul | `(3,4) @ (4,5)` | Check gradients for both inputs |
| Add | `(3,4) + (3,4)` | Also test broadcasting `(3,4) + (4)` |
| Multiply | `(3,4) * (3,4)` | Element-wise |
| Softmax | `(3,5)` | Check along dim=1 |
| ReLU | `(3,4)` | Include some negative values |
| LayerNormOp | `(2,3,4)` | Check gamma/beta gradients too |
| CrossEntropy | `(3,5)` logits, 3 targets | Combined with softmax |
| Transpose | `(3,4)` | Swap dims 0,1 |
| EmbeddingLookup | `(5,4)` weight, 3 indices | Only indexed rows get gradient |
| Mask | `(3,4)` | Masked positions get zero gradient |

**Integration in Demos:** The demo programs can optionally run gradient checks on model sub-components before training starts (enabled by a flag/argument). This lets the learner verify correctness before committing to a long training run.

### 9.2 Training Validation (Behavioral Testing)

These are not automated unit tests but rather observable behaviors:

| Validation | What to Observe |
|------------|-----------------|
| **Loss decreases** | Print loss every N steps; should trend downward |
| **Overfitting on tiny data** | Train on 1-2 sentences; model should memorize them perfectly. If it can't, something is broken. |
| **Text quality improves** | Early samples are gibberish; later samples are English-like |
| **Reversal accuracy improves** | Encoder-decoder should go from 0% to 80%+ on short strings |

### 9.3 Determinism

A `Random` seed is threaded through all random operations (weight initialization, sampling, batch shuffling). Setting the same seed produces the same results, making debugging reproducible.

---

## 10. Implementation Plan

### Phase 1: Autograd Engine

**Components:** `Tensor`, `Operation`, all ops in `autograd.ops/`, `GradientChecker`

**Why first:** Everything else depends on this. The tensor operations must be correct before building layers on top.

**Acceptance Criteria:**
- [ ] Tensor class supports N-dimensional shape, row-major flat storage, gradient tracking
- [ ] All core operations implemented: MatMul, Add, Multiply, Scale, Softmax, ReLU, Transpose, Reshape, LayerNormOp, CrossEntropy, EmbeddingLookup, Mask
- [ ] `backward()` performs correct topological sort and reverse-mode autodiff
- [ ] GradientChecker passes for every operation with relative error < 1e-4
- [ ] Gradients accumulate correctly when a tensor is used in multiple operations
- [ ] `zeroGrad()` properly resets all gradients

**Estimated Effort:** Largest phase. The MatMul and LayerNormOp backward passes are the most complex. Most other ops are straightforward.

**Dependencies:** None.

### Phase 2: Transformer Layers

**Components:** `Linear`, `Embedding`, `PositionalEncoding`, `LayerNorm`, `FeedForward`, `MultiHeadAttention`

**Why second:** These compose autograd ops into meaningful Transformer building blocks. Each can be gradient-checked independently.

**Acceptance Criteria:**
- [ ] Linear produces correct output shapes; gradient flows through weight and bias
- [ ] Embedding maps indices to vectors; gradient flows only to selected rows
- [ ] PositionalEncoding adds correct sinusoidal values; verified against hand-computed examples
- [ ] LayerNorm normalizes correctly; gamma/beta gradients are correct
- [ ] FeedForward composes Linear + ReLU + Linear correctly
- [ ] MultiHeadAttention correctly splits heads, computes scaled dot-product attention, concatenates, projects
- [ ] MultiHeadAttention works with and without causal mask
- [ ] MultiHeadAttention works in cross-attention mode (separate query and key/value sources)
- [ ] All layers pass gradient checking via GradientChecker

**Dependencies:** Phase 1 (Autograd Engine)

### Phase 3: Model Configurations

**Components:** `TransformerBlock`, `DecoderOnlyModel`, `EncoderDecoderModel`

**Why third:** These compose the layers into complete models. Needs working layers.

**Acceptance Criteria:**
- [ ] TransformerBlock correctly combines attention + FFN with pre-norm and residual connections
- [ ] TransformerBlock configurable for encoder (no mask, no cross-attn), decoder (mask, no cross-attn), and full decoder (mask + cross-attn)
- [ ] DecoderOnlyModel produces logits of shape `(batch, seqLen, vocabSize)`
- [ ] DecoderOnlyModel uses weight tying between embedding and output projection
- [ ] EncoderDecoderModel produces logits from source and target token sequences
- [ ] Both models expose `getParameters()` returning all learnable tensors
- [ ] A single forward + backward pass runs without errors on small random inputs

**Dependencies:** Phase 2 (Transformer Layers)

### Phase 4: Data Pipeline and Training

**Components:** `CharTokenizer`, `TextReader`, `SequenceBatcher`, `ReversalTaskGenerator`, `AdamOptimizer`

**Why fourth:** The models need data and an optimizer to train. These are relatively straightforward.

**Acceptance Criteria:**
- [ ] CharTokenizer correctly encodes and decodes strings (roundtrip test)
- [ ] CharTokenizer supports special tokens for encoder-decoder tasks
- [ ] TextReader reads Shakespeare file and splits train/validation
- [ ] SequenceBatcher produces correctly offset input/target pairs
- [ ] ReversalTaskGenerator produces valid reversal pairs with proper SOS/EOS/PAD tokens
- [ ] AdamOptimizer correctly updates parameters (verify on a simple convex problem: minimize x^2)
- [ ] Training loop reduces loss when overfitting on tiny data

**Dependencies:** Phase 3 (Models), Shakespeare text file in `data/`

### Phase 5: Demo Programs and Polish

**Components:** `CharGenerationTest`, `TranslationTest`, interactive generation, model save/load

**Why last:** End-to-end integration. Requires all other components.

**Acceptance Criteria:**
- [ ] `CharGenerationTest` compiles and runs end-to-end
- [ ] Loss visibly decreases during Shakespeare training
- [ ] Generated text improves from gibberish to recognizable English fragments
- [ ] Interactive mode reads prompts from stdin and generates continuations
- [ ] Temperature parameter visibly affects generation randomness
- [ ] `TranslationTest` compiles and runs end-to-end
- [ ] String reversal accuracy reaches >= 80% on short strings (length <= 10)
- [ ] Model save/load works via plain text format (save after training, load and generate)
- [ ] Saved model file is human-readable (can open in a text editor and see named tensors)
- [ ] Training completes in minutes to a few hours (not days)
- [ ] All code compiles with `javac -d out $(find src -name "*.java")`

**Dependencies:** All previous phases.

---

## 11. Risks and Mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | **Autograd gradient bugs** --- subtle errors in backward passes cause silent training failure | Medium | High | GradientChecker validates every operation numerically. Test each op in isolation before composing. |
| 2 | **Performance too slow** --- pure Java matmul on CPU may be prohibitively slow for batched 3D operations | Medium | Medium | Keep model small (~300K params). Use flat `double[]` arrays and direct index math (no object allocation in inner loops). Profile matmul if needed --- it's the bottleneck. |
| 3 | **Numerical instability** --- softmax overflow, gradient explosion, NaN propagation | Medium | Medium | Numerically stable softmax (subtract max). Pre-norm architecture (more stable than post-norm). Small learning rate (3e-4). Monitor for NaN in training loop and halt early with diagnostic message. |
| 4 | **Shape mismatches** --- wrong tensor shapes in attention head splitting/merging or batched matmul | Medium | Low | Add shape assertions in forward methods. Clear error messages showing expected vs actual shapes. |
| 5 | **Memory overhead from computation graph** --- storing every intermediate tensor for backward pass | Low | Medium | Model is small. If needed, implement a `no_grad()` context for inference that skips graph construction. |
| 6 | **Autograd complexity obscures learning** --- learner gets lost in the graph machinery | Low | Medium | Thorough documentation. GradientChecker as a verification tool. Each Operation class is self-contained with its own backward logic clearly commented. |

---

## 12. Appendices

### A. Glossary

See REQUIREMENTS.md Appendix A for the complete glossary. Additional specification-specific terms:

| Term | Definition |
|------|-----------|
| **Row-major order** | Array storage where the last index varies fastest. Standard C/Java convention. |
| **Stride** | The number of elements to skip in the flat array to move one step along a dimension. |
| **Pre-norm** | Applying LayerNorm before (rather than after) each sublayer. More stable. Used in GPT-2+. |
| **Weight tying** | Sharing the embedding weight matrix between input embedding and output projection. |
| **Teacher forcing** | Training the decoder by feeding it the correct previous token (not its own prediction). Standard for sequence-to-sequence training. |
| **Xavier/Glorot initialization** | Weight initialization scaled by `sqrt(2/(fan_in + fan_out))`. Prevents vanishing/exploding signals. |

### B. Model Parameter Counts

**Decoder-Only Model (Shakespeare):**
```
embedDim=128, numLayers=4, numHeads=4, ffnInnerDim=512, vocabSize≈65

Embedding:          65 * 128                    =     8,320
Per TransformerBlock:
  LayerNorm (x2):   128 * 2 * 2                =       512
  MHA (W_Q,K,V,O):  128 * 128 * 4 + 128 * 4   =    66,048
  FFN (linear1+2):  128 * 512 + 512 + 512 * 128 + 128 = 131,712
  Block total:                                  ≈   198,272
Blocks (x4):                                    ≈   793,088
Final LayerNorm:    128 * 2                     =       256
Output projection:  (tied with embedding)       =         0
                                                ----------
Total:                                          ≈   801,664
```

At ~800K parameters, this is still very small. Training time on CPU should be well within the target of a few hours. If it's too slow, reducing to `embedDim=64, numLayers=2` cuts parameters by ~8x.

**Encoder-Decoder Model (Reversal):**
```
embedDim=64, numLayers=2, numHeads=2, ffnInnerDim=256, vocabSize≈30

Source Embedding:     30 * 64                   =     1,920
Target Embedding:     30 * 64                   =     1,920
Encoder Block (x2):   ~50K each                 ≈   100,000
Decoder Block (x2):   ~75K each (has cross-attn)≈   150,000
Final LayerNorm:      64 * 2                    =       128
Output Projection:    64 * 30 + 30              =     1,950
                                                ----------
Total:                                          ≈   255,918
```

~256K parameters. Trivial on CPU.

### C. Hyperparameter Defaults Summary

| Parameter | CharGenerationTest | TranslationTest |
|-----------|-------------------|-----------------|
| `embedDim` | 128 | 64 |
| `numLayers` | 4 | 2 |
| `numHeads` | 4 | 2 |
| `ffnInnerDim` | 512 | 256 |
| `maxSeqLen` | 128 | 32 |
| `batchSize` | 32 | 64 |
| `learningRate` | 3e-4 | 1e-3 |
| `beta1` | 0.9 | 0.9 |
| `beta2` | 0.999 | 0.999 |
| `epsilon` | 1e-8 | 1e-8 |
| `temperature` (generation) | 0.8 | N/A (greedy) |
| `maxGenerateLen` | 200 | sequence length |
| `randomSeed` | 42 | 42 |

### D. Key References

1. Vaswani, A. et al. (2017). "Attention Is All You Need." *arXiv:1706.03762*
2. Kingma, D.P. & Ba, J. (2014). "Adam: A Method for Stochastic Optimization." *arXiv:1412.6980*
3. Ba, J.L. et al. (2016). "Layer Normalization." *arXiv:1607.06450*
4. Karpathy, A. "nanoGPT" --- conceptual reference for minimal GPT implementation
5. Glorot, X. & Bengio, Y. (2010). "Understanding the difficulty of training deep feedforward neural networks." --- Xavier initialization
