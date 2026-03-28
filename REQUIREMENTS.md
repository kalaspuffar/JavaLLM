# Requirements Document: JavaLLM

**Version:** 1.0
**Date:** 2026-03-28
**Author:** Requirements Analyst (Claude)
**Stakeholder:** Repository Owner
**Status:** Approved by stakeholder

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Business Context](#2-business-context)
3. [Goals and Objectives](#3-goals-and-objectives)
4. [Scope](#4-scope)
5. [Stakeholders](#5-stakeholders)
6. [User Personas](#6-user-personas)
7. [Functional Requirements](#7-functional-requirements)
8. [Non-Functional Requirements](#8-non-functional-requirements)
9. [Data Requirements](#9-data-requirements)
10. [Integration Requirements](#10-integration-requirements)
11. [Constraints](#11-constraints)
12. [Assumptions](#12-assumptions)
13. [Dependencies](#13-dependencies)
14. [Risks](#14-risks)
15. [Success Criteria](#15-success-criteria)
16. [Open Questions](#16-open-questions)
17. [Appendices](#17-appendices)

---

## 1. Executive Summary

JavaLLM is an educational, from-scratch implementation of the Transformer architecture in pure Java. It follows the same philosophy as the existing JavaCNN project in this repository: **build every piece by hand to understand how it works**.

The project will implement the core building blocks of a Transformer — embeddings, positional encoding, self-attention, multi-head attention, feed-forward networks, and layer normalization — along with a small automatic differentiation engine. These components will be composed into two model configurations:

1. **Encoder-Decoder Transformer** — demonstrated with a simple character-level task (e.g., string reversal)
2. **Decoder-Only Transformer** — demonstrated with character-level Shakespeare text generation

The primary success criterion is **clarity**: a developer reading the code should be able to understand how and why each piece of a Transformer works, and see how the decoder-only architecture used by modern LLMs is a simplification of the full original design.

---

## 2. Business Context

### Background and Rationale

The repository owner previously built JavaCNN — a pure Java port of ConvNetJS — as a hands-on way to understand convolutional neural networks. That project successfully demystified CNNs by implementing every layer, every gradient calculation, and the full training pipeline from scratch.

The same learning approach is now desired for **Large Language Models (LLMs)**. LLMs are built on the Transformer architecture introduced in "Attention Is All You Need" (Vaswani et al., 2017). While countless tutorials exist, few offer a complete, self-contained implementation in a mainstream language without framework dependencies.

### Current State

- `JavaCNN/` — Complete, working CNN implementation. Trains on MNIST, classifies digits.
- `JavaLLM/` — Empty directory, ready for this project.

### Desired State

- `JavaLLM/` — Complete, working Transformer implementation. Trains a character-level model on Shakespeare text. Generates new text from prompts. Includes both encoder-decoder and decoder-only configurations with shared building blocks.

### Strategic Alignment

This project fits the repository's purpose: **a personal learning workspace for understanding neural network architectures through implementation**.

---

## 3. Goals and Objectives

### Primary Goal

**Understand the Transformer architecture by building it from scratch in Java.**

### Specific Objectives

| # | Objective | Measurable Outcome |
|---|-----------|-------------------|
| 1 | Understand token embeddings and positional encoding | Working Embedding and PositionalEncoding classes with clear forward/backward |
| 2 | Understand the self-attention mechanism | Working SelfAttention class; can inspect attention weights |
| 3 | Understand multi-head attention | Working MultiHeadAttention that composes multiple attention heads |
| 4 | Understand the full Transformer block | Working TransformerBlock combining attention + feed-forward + layer norm |
| 5 | Understand encoder-decoder vs decoder-only | Both model types built from the same shared components |
| 6 | Understand autograd / computation graphs | Working automatic differentiation engine; understand how PyTorch works under the hood |
| 7 | See a model train and improve | Watch loss decrease; see generated text go from gibberish to recognizable English |
| 8 | Experience interactive inference | Type a prompt, get generated text back |

### Key Performance Indicators

- **Training time**: Full training run completes in **under 1 hour** on CPU
- **Learning progression**: Visible improvement in generated text quality across training
- **Code readability**: Any Java developer can read a single class and understand what it does without cross-referencing 10 other files

---

## 4. Scope

### In Scope

**Core Autograd Engine**
- Tensor class with automatic gradient tracking
- Computation graph construction during forward pass
- Backward pass (reverse-mode automatic differentiation)
- Core operations: matmul, add, element-wise multiply, transpose, softmax, tanh, ReLU, layer norm, cross-entropy loss

**Transformer Building Blocks**
- Token Embedding layer
- Positional Encoding (sinusoidal, from the original paper)
- Scaled Dot-Product Attention
- Multi-Head Attention
- Position-wise Feed-Forward Network (two linear layers with activation)
- Layer Normalization
- Residual connections
- Causal masking (for decoder self-attention)

**Model Configurations**
- Encoder-Decoder Transformer (full original architecture)
- Decoder-Only Transformer (GPT-style, for text generation)

**Data Pipeline**
- Character-level tokenizer (char-to-index, index-to-char)
- Text file reader for Shakespeare corpus
- Sequence batching (split text into fixed-length training windows)
- Synthetic data generator for encoder-decoder task (e.g., string reversal)

**Training Infrastructure**
- Adam optimizer (the standard for Transformer training)
- Training loop with loss reporting
- Periodic text generation samples during training (to visualize progress)
- Model save/load (Java serialization, matching JavaCNN's approach)

**Inference**
- Autoregressive text generation (one token at a time)
- Temperature-controlled sampling
- Interactive prompt mode (read from stdin, generate continuation)

**Demo Programs**
- `CharGenerationTest` — Decoder-only: train on Shakespeare, generate text
- `TranslationTest` — Encoder-decoder: train on a simple character-level task

### Out of Scope

- GPU acceleration / CUDA
- External libraries (no ND4J, DL4J, etc.)
- Byte-pair encoding (BPE) or other subword tokenizers
- Beam search decoding
- Distributed training
- Quantization or model compression
- Production-grade performance optimization
- Pre-trained model weights
- Web UI or REST API
- Attention visualization (could be a future phase)

### Future Considerations (Later Phases)

- BPE tokenizer (understand how real LLMs tokenize)
- Attention weight visualization (heatmaps showing what attends to what)
- Larger training corpus or word-level model
- Key-value caching for faster inference
- Learning rate scheduling (warmup + decay)
- Gradient clipping
- Additional optimizer variants
- Beam search or top-k/top-p sampling refinements

---

## 5. Stakeholders

| Stakeholder | Role | Interest |
|-------------|------|----------|
| Repository Owner | Learner, Developer | Understand Transformers hands-on; have readable reference code |
| Future Readers | Secondary audience | Anyone who reads the repo to learn about Transformers |

---

## 6. User Personas

### Primary: The Hands-On Learner

- **Who**: A developer who understands Java and basic ML concepts (has built a CNN)
- **Goal**: Understand how Transformers and LLMs work at the implementation level
- **Technical level**: Strong Java developer; understands forward/backward pass, gradient descent, loss functions from the CNN project; new to attention mechanisms and Transformers
- **Usage pattern**: Reads code, modifies parameters, runs training, observes results, experiments

### Secondary: The Code Reader

- **Who**: Any developer browsing the repository
- **Goal**: Read the code to understand Transformer internals
- **Technical level**: Varies; code should be self-explanatory with good naming and comments
- **Usage pattern**: Reads source files, follows the data flow, understands the architecture

---

## 7. Functional Requirements

### 7.1 Autograd Engine

The autograd engine is itself an educational component. It should be clearly documented and understandable.

#### FR-AG-1: Tensor Class
- **Priority**: Must-have
- A `Tensor` class that wraps a multi-dimensional array of doubles
- Stores shape information (dimensions)
- Optionally tracks gradients (`requiresGrad` flag)
- Holds a reference to the operation that created it (for graph traversal)

#### FR-AG-2: Computation Graph
- **Priority**: Must-have
- Operations on tensors build a directed acyclic graph (DAG) automatically
- Each node records its inputs and the operation performed
- Graph is built dynamically during the forward pass (define-by-run, like PyTorch)

#### FR-AG-3: Backward Pass
- **Priority**: Must-have
- Calling `backward()` on a scalar loss tensor triggers reverse-mode autodiff
- Gradients accumulate in each tensor's `.grad` field
- Topological sort ensures correct ordering
- Gradients can be zeroed for the next training step

#### FR-AG-4: Core Operations
- **Priority**: Must-have
- Matrix multiplication (matmul)
- Element-wise addition, subtraction, multiplication
- Transpose and reshape
- Softmax (numerically stable)
- ReLU and/or GELU activation
- Layer normalization
- Cross-entropy loss
- Embedding lookup (index-based selection of rows)
- Masking (applying -infinity to attention scores)

### 7.2 Transformer Building Blocks

Each building block should be its own class with a clear `forward()` method. Following the JavaCNN pattern of one class per layer type.

#### FR-TB-1: Token Embedding
- **Priority**: Must-have
- Maps integer token IDs to dense vectors
- Learnable weight matrix of shape (vocab_size, embed_dim)
- Forward: look up rows by token index
- Backward: gradient flows only to the looked-up rows

#### FR-TB-2: Positional Encoding
- **Priority**: Must-have
- Adds position information to embeddings
- Use sinusoidal encoding from "Attention Is All You Need"
- Fixed (not learned) — computed once based on position and dimension
- Forward: add positional encoding to the input embeddings

#### FR-TB-3: Scaled Dot-Product Attention
- **Priority**: Must-have
- Implements: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`
- Accepts an optional mask (for causal / padding masking)
- This is the core mechanism — should be especially well-documented

#### FR-TB-4: Multi-Head Attention
- **Priority**: Must-have
- Splits input into multiple heads, applies attention to each, concatenates results
- Learnable projection matrices: W_Q, W_K, W_V, W_O
- Number of heads is configurable

#### FR-TB-5: Position-wise Feed-Forward Network
- **Priority**: Must-have
- Two linear transformations with an activation in between
- `FFN(x) = W2 * activation(W1 * x + b1) + b2`
- Inner dimension is typically 4x the model dimension

#### FR-TB-6: Layer Normalization
- **Priority**: Must-have
- Normalizes across the feature dimension
- Learnable scale (gamma) and shift (beta) parameters
- Applied before or after attention/FFN (pre-norm is simpler and more stable)

#### FR-TB-7: Residual Connections
- **Priority**: Must-have
- `output = LayerNorm(x + Sublayer(x))` pattern
- Present around both attention and feed-forward sublayers

#### FR-TB-8: Causal Mask
- **Priority**: Must-have
- Upper-triangular mask that prevents attending to future positions
- Used in decoder self-attention
- Fills masked positions with negative infinity before softmax

### 7.3 Model Configurations

#### FR-MC-1: Transformer Block (Shared)
- **Priority**: Must-have
- Composes: Multi-Head Attention + Feed-Forward + Layer Norm + Residual Connections
- Configurable as encoder block (no causal mask) or decoder block (with causal mask)
- Decoder block optionally includes cross-attention to encoder output

#### FR-MC-2: Encoder-Decoder Transformer
- **Priority**: Must-have
- Stack of N encoder blocks processing the input sequence
- Stack of N decoder blocks generating the output sequence
- Decoder blocks include cross-attention to the encoder's output
- Separate input/output embeddings (may share vocabulary)
- Final linear projection + softmax to produce output token probabilities

#### FR-MC-3: Decoder-Only Transformer
- **Priority**: Must-have
- Stack of N decoder blocks with causal (masked) self-attention
- No encoder, no cross-attention
- Single embedding layer (shared input/output)
- Final linear projection + softmax for next-token prediction
- Should be clearly shown as a **simplification** of the full architecture

### 7.4 Data Pipeline

#### FR-DP-1: Character Tokenizer
- **Priority**: Must-have
- Scans training text to build vocabulary (unique characters)
- Maps characters to integer IDs and back
- Handles a small, fixed vocabulary (printable ASCII from the corpus)
- Includes special tokens if needed (e.g., padding, start/end for encoder-decoder)

#### FR-DP-2: Text File Reader
- **Priority**: Must-have
- Reads a plain text file (Shakespeare corpus) into memory
- Splits into training and optional validation portions
- No complex preprocessing beyond tokenization

#### FR-DP-3: Sequence Batcher
- **Priority**: Must-have
- Splits tokenized text into fixed-length sequences (context windows)
- For decoder-only: input = tokens[0..n-1], target = tokens[1..n]
- For encoder-decoder: input/output pairs from the task generator
- Produces batches of multiple sequences for efficient training

#### FR-DP-4: Synthetic Task Generator (Encoder-Decoder)
- **Priority**: Must-have
- Generates input-output pairs for a simple character-level task
- Example tasks: string reversal, character shifting, simple substitution
- Provides clear, verifiable correctness (easy to see if the model got it right)

### 7.5 Training Infrastructure

#### FR-TI-1: Adam Optimizer
- **Priority**: Must-have
- Standard Adam algorithm (Kingma & Ba, 2014)
- Maintains first and second moment estimates per parameter
- Configurable learning rate, beta1, beta2, epsilon
- Iterates over all model parameters and applies updates using accumulated gradients

#### FR-TI-2: Training Loop
- **Priority**: Must-have
- For each epoch: iterate over batches, forward pass, compute loss, backward pass, optimizer step, zero gradients
- Print loss at configurable intervals
- Generate sample text at configurable intervals (decoder-only)
- Report training speed (tokens/second or batches/second)

#### FR-TI-3: Model Persistence
- **Priority**: Should-have
- Save trained model to disk (Java serialization, matching JavaCNN approach)
- Load model from disk for continued training or inference
- Save/load optimizer state for training resumption

#### FR-TI-4: Progress Reporting
- **Priority**: Must-have
- Print current epoch, batch, loss at regular intervals
- For decoder-only: periodically generate and print a text sample to show learning progress
- For encoder-decoder: periodically run a few test inputs and print predicted vs expected output

### 7.6 Inference

#### FR-IN-1: Autoregressive Generation
- **Priority**: Must-have
- Given a prompt (sequence of characters), generate continuation one token at a time
- Feed each generated token back as input for the next step
- Stop after a configurable maximum length

#### FR-IN-2: Temperature Sampling
- **Priority**: Must-have
- Temperature parameter controls randomness of generation
- Temperature = 1.0: standard sampling from the probability distribution
- Temperature < 1.0: more deterministic (sharper distribution)
- Temperature > 1.0: more random (flatter distribution)

#### FR-IN-3: Interactive Mode
- **Priority**: Should-have
- Read prompts from stdin in a loop
- Generate and print continuation for each prompt
- Allow user to set temperature and generation length

### 7.7 Demo Programs

#### FR-DM-1: CharGenerationTest
- **Priority**: Must-have
- End-to-end demo of decoder-only Transformer
- Downloads or reads Shakespeare text from a bundled file
- Builds a decoder-only Transformer with configurable hyperparameters
- Trains the model, printing loss and text samples
- After training, enters interactive generation mode
- Should be runnable with a single `java` command

#### FR-DM-2: TranslationTest
- **Priority**: Must-have
- End-to-end demo of encoder-decoder Transformer
- Generates synthetic training data (e.g., string reversal)
- Builds an encoder-decoder Transformer
- Trains the model, printing loss and example predictions
- After training, tests on new inputs to demonstrate generalization

---

## 8. Non-Functional Requirements

### 8.1 Performance

| Requirement | Target |
|-------------|--------|
| **Training time** (decoder-only, Shakespeare) | Under 1 hour on a modern CPU |
| **Text generation** | Interactive speed — under 5 seconds for a 200-character continuation |
| **Memory usage** | Fits comfortably in default JVM heap (256MB-1GB) |

To achieve under-1-hour training, the model should be small:
- Embedding dimension: 64-128
- Number of layers: 2-4
- Number of attention heads: 2-4
- Context window: 64-128 characters
- Training corpus: ~100KB-1MB of text

These are tuning suggestions, not hard requirements. The architect should size the model to meet the training time target.

### 8.2 Readability (Critical)

**This is the most important non-functional requirement.**

| Requirement | Detail |
|-------------|--------|
| **One concept per class** | Each class implements one clear idea |
| **Self-documenting names** | Variable and method names explain their purpose |
| **Comments explain "why"** | Not what the code does, but why it does it that way |
| **Reference to theory** | Key classes should comment which part of "Attention Is All You Need" they implement |
| **No clever tricks** | Prefer obvious code over optimized code |
| **Consistent patterns** | Follow the same forward/backward pattern throughout, matching JavaCNN's style |

### 8.3 Correctness

| Requirement | Detail |
|-------------|--------|
| **Gradient checking** | Autograd engine should include a numerical gradient check utility for verification |
| **Known-good behavior** | Loss should decrease during training; generated text should visibly improve |
| **Deterministic option** | Ability to set a random seed for reproducible results |

### 8.4 Usability

| Requirement | Detail |
|-------------|--------|
| **Simple compilation** | `javac -d out $(find src -name "*.java")` — same as JavaCNN |
| **Simple execution** | `java -cp out org.ea.javallm.CharGenerationTest` |
| **No external dependencies** | No jars, no downloads, no build tools required |
| **Bundled training data** | Include a small Shakespeare text file in the repo (public domain) |
| **Sensible defaults** | Runs out of the box with default hyperparameters |

### 8.5 Maintainability

| Requirement | Detail |
|-------------|--------|
| **Modular design** | Components can be understood, modified, and tested independently |
| **Extensible** | Easy to add new layer types, optimizers, or tasks in the future |
| **Match JavaCNN patterns** | Similar project structure, similar interface patterns, familiar feel |

---

## 9. Data Requirements

### 9.1 Training Data — Shakespeare Corpus

| Attribute | Detail |
|-----------|--------|
| **Source** | Tiny Shakespeare dataset (public domain) |
| **Format** | Plain text file, UTF-8 |
| **Size** | ~100KB to ~1MB |
| **Location** | Bundled in the project (e.g., `JavaLLM/data/shakespeare.txt`) |
| **Preprocessing** | None required beyond character tokenization |
| **Train/validation split** | Last 10% of text for validation, first 90% for training |

### 9.2 Synthetic Data — Encoder-Decoder Tasks

| Attribute | Detail |
|-----------|--------|
| **Source** | Generated programmatically at runtime |
| **Format** | Pairs of character sequences (input, expected output) |
| **Size** | Thousands of pairs, generated on demand |
| **Example task** | String reversal: "hello" → "olleh" |

### 9.3 Vocabulary

| Attribute | Detail |
|-----------|--------|
| **Type** | Character-level |
| **Size** | ~65-95 unique characters (depends on corpus) |
| **Construction** | Scan training text, assign sequential IDs |
| **Special tokens** | Padding, start-of-sequence, end-of-sequence (as needed for encoder-decoder) |

---

## 10. Integration Requirements

**None.** This is a fully self-contained, standalone project. No external services, APIs, or systems to integrate with.

The project sits alongside JavaCNN in the same repository but is completely independent.

---

## 11. Constraints

### Technical Constraints

| Constraint | Detail |
|------------|--------|
| **Language** | Java (JDK 17+; JDK 21 is available on the target system) |
| **No external libraries** | Everything built from scratch — no ND4J, DL4J, Deeplearning4j, or any ML framework |
| **No GPU** | CPU-only execution |
| **No build tool required** | Must compile with plain `javac`; Maven may be added later but is not required |
| **Computation** | All numerical operations use Java `double` arrays |

### Design Constraints

| Constraint | Detail |
|------------|--------|
| **Readability over performance** | Always choose the clearer implementation |
| **Match JavaCNN conventions** | Similar project structure, package organization, interface patterns |
| **One class per file** | Standard Java convention |
| **Package prefix** | `org.ea.javallm` (matching `org.ea.javacnn`) |

### Resource Constraints

| Constraint | Detail |
|------------|--------|
| **Training time** | Must complete in under 1 hour |
| **Memory** | Must run within standard JVM heap settings |
| **No special hardware** | Standard desktop/laptop CPU |

---

## 12. Assumptions

| # | Assumption |
|---|-----------|
| 1 | JDK 17 or newer is available (JDK 21 confirmed on target system) |
| 2 | The Tiny Shakespeare text is small enough to fit entirely in memory |
| 3 | Character-level tokenization provides sufficient granularity for a learning exercise |
| 4 | `double` precision is adequate for this educational model (no need for float16/bfloat16) |
| 5 | Java's math performance is sufficient for the target model size and training time |
| 6 | The learner has already built JavaCNN and understands forward/backward passes, gradient descent, and loss functions |
| 7 | Sinusoidal positional encoding is sufficient (no need for learned or rotary encodings) |
| 8 | A single optimizer (Adam) is sufficient for the initial implementation |
| 9 | Serialization via Java's built-in mechanism is acceptable for model save/load |

---

## 13. Dependencies

| # | Dependency | Type | Status |
|---|-----------|------|--------|
| 1 | JDK 17+ | Runtime | Available (JDK 21 installed) |
| 2 | Shakespeare text corpus | Training data | Must be obtained (public domain, freely available) |
| 3 | Understanding of JavaCNN architecture | Knowledge | Completed |

No external software dependencies. No vendor dependencies. No team dependencies.

---

## 14. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | **Autograd complexity** — building a correct autograd engine is non-trivial; subtle bugs in gradient computation are hard to spot | Medium | High | Include numerical gradient checking utility; test each operation individually |
| 2 | **Performance** — pure Java double-precision math on CPU may be slow for matrix operations | Medium | Medium | Keep model very small; optimize inner loops with direct array access (same approach as JavaCNN); profile if needed |
| 3 | **Training time exceeds 1 hour** | Medium | Low | Reduce model size, context window, or training epochs; this is a tuning parameter, not an architectural issue |
| 4 | **Autograd approach may be too abstract** — learner may find it harder to understand than manual backprop | Low | Medium | Document the autograd engine thoroughly; include option to trace/print computation graph; can always add manual backprop for specific layers later |
| 5 | **Numerical instability** — softmax overflow, vanishing/exploding gradients | Medium | Medium | Use numerically stable softmax (subtract max); use pre-norm (LayerNorm before attention); keep model small |
| 6 | **Scope creep** — temptation to add more features | Low | Low | Strict adherence to in-scope list; future phases clearly defined |

---

## 15. Success Criteria

### Minimum Viable Success

The project is successful when:

1. **It compiles and runs** with `javac` + `java`, no external dependencies
2. **Loss decreases** during training, demonstrating that the model is learning
3. **Generated text improves visibly** from random characters to recognizable English-like output
4. **The encoder-decoder model** can learn a simple character-level task (e.g., ≥80% accuracy on string reversal for short strings)
5. **Interactive generation works** — user types a prompt, model produces a continuation
6. **The code is readable** — each class can be understood in isolation with minimal cross-referencing

### Stretch Goals

- Generated Shakespeare text that is grammatically plausible (even if nonsensical)
- Attention weights can be printed/inspected to see what the model attends to
- Training loss curves can be logged to a file for plotting

### Acceptance Test

```
# Compile
cd JavaLLM
javac -d out $(find src -name "*.java")

# Train decoder-only model on Shakespeare (should complete in < 1 hour)
java -cp out org.ea.javallm.CharGenerationTest

# Observe: loss decreasing, text samples improving over training
# After training: interactive prompt generates text continuations

# Train encoder-decoder model on string reversal
java -cp out org.ea.javallm.TranslationTest

# Observe: loss decreasing, accuracy improving, model reverses strings correctly
```

---

## 16. Open Questions

| # | Question | Impact | Suggested Resolution |
|---|----------|--------|---------------------|
| 1 | **Exact encoder-decoder task** — string reversal is simple but are there more interesting options? | Low | Start with string reversal; can swap in another task later. The architecture is task-agnostic. |
| 2 | **Pre-norm vs post-norm** — the original paper uses post-norm (LayerNorm after residual add) but pre-norm (LayerNorm before sublayer) is more stable and used in modern models. Which to implement? | Low | Implement pre-norm (simpler, more stable). Comment explaining the difference. |
| 3 | **Weight tying** — should the decoder output projection share weights with the input embedding? | Low | Yes, implement weight tying. It's standard practice, reduces parameters, and is a useful concept to understand. |
| 4 | **Activation function** — ReLU (original paper) or GELU (used in GPT-2+)? | Low | Implement ReLU for simplicity. Mention GELU in comments as a modern alternative. |
| 5 | **Batch size** — what's practical for CPU training? | Low | Architect should determine through testing. Likely 16-64 sequences per batch. |

---

## 17. Appendices

### A. Glossary

| Term | Definition |
|------|-----------|
| **Transformer** | Neural network architecture based on self-attention, introduced in "Attention Is All You Need" (2017) |
| **Self-Attention** | Mechanism where each position in a sequence attends to all other positions to compute its representation |
| **Multi-Head Attention** | Running multiple attention operations in parallel, each with different learned projections |
| **Causal Mask** | Mask that prevents a position from attending to future positions (used in text generation) |
| **Cross-Attention** | Attention where queries come from the decoder and keys/values come from the encoder |
| **Autograd** | Automatic differentiation — automatically computing gradients by recording operations |
| **Computation Graph** | DAG of operations built during the forward pass, traversed in reverse during backward pass |
| **Embedding** | Mapping from discrete tokens (integers) to continuous vectors |
| **Positional Encoding** | Signal added to embeddings to convey position information (since attention has no inherent sense of order) |
| **Layer Normalization** | Normalizing activations across the feature dimension to stabilize training |
| **Residual Connection** | Adding the input of a sublayer to its output: `output = x + sublayer(x)` |
| **Adam** | Adaptive Moment Estimation optimizer; maintains per-parameter learning rates |
| **Temperature** | Scaling factor applied to logits before sampling; controls randomness of generation |
| **Autoregressive** | Generating one token at a time, feeding each generated token back as input |
| **Context Window** | The fixed number of tokens the model can attend to at once |

### B. Reference Architecture Diagram

```
ENCODER-DECODER TRANSFORMER
============================

Input Text          Output Text (shifted right)
    |                        |
[Char Tokenizer]      [Char Tokenizer]
    |                        |
[Token Embedding]     [Token Embedding]
    +                        +
[Positional Enc.]     [Positional Enc.]
    |                        |
    v                        v
+-------------------+  +------------------------+
| ENCODER BLOCK x N |  | DECODER BLOCK x N      |
|                   |  |                         |
| [Multi-Head       |  | [Masked Multi-Head      |
|  Self-Attention]  |  |  Self-Attention]        |
|       |           |  |       |                  |
| [Add & LayerNorm] |  | [Add & LayerNorm]       |
|       |           |  |       |                  |
| [Feed-Forward]    |  | [Multi-Head             |
|       |           |  |  Cross-Attention] <------+-- Encoder Output
| [Add & LayerNorm] |  |       |                  |
|                   |  | [Add & LayerNorm]       |
+-------------------+  |       |                  |
                        | [Feed-Forward]          |
                        |       |                  |
                        | [Add & LayerNorm]       |
                        +------------------------+
                                |
                        [Linear Projection]
                                |
                        [Softmax]
                                |
                        Next Token Probabilities


DECODER-ONLY TRANSFORMER (GPT-style)
======================================

Input Text
    |
[Char Tokenizer]
    |
[Token Embedding]
    +
[Positional Enc.]
    |
    v
+------------------------+
| DECODER BLOCK x N      |
|                         |
| [Masked Multi-Head      |
|  Self-Attention]        |
|       |                  |
| [Add & LayerNorm]       |
|       |                  |
| [Feed-Forward]          |
|       |                  |
| [Add & LayerNorm]       |
+------------------------+
         |
 [Linear Projection]
         |
 [Softmax]
         |
 Next Token Probabilities
```

### C. Suggested Project Structure

```
JavaLLM/
├── data/
│   └── shakespeare.txt              # Training corpus
├── src/org/ea/javallm/
│   ├── autograd/
│   │   ├── Tensor.java              # Core tensor with gradient tracking
│   │   ├── Operation.java           # Base class for differentiable operations
│   │   ├── ops/
│   │   │   ├── MatMul.java          # Matrix multiplication
│   │   │   ├── Add.java             # Element-wise addition
│   │   │   ├── Softmax.java         # Softmax with stable computation
│   │   │   ├── ReLU.java            # ReLU activation
│   │   │   ├── LayerNormOp.java     # Layer normalization operation
│   │   │   ├── CrossEntropy.java    # Cross-entropy loss
│   │   │   ├── Reshape.java         # Reshape / view
│   │   │   └── Transpose.java       # Matrix transpose
│   │   └── GradientChecker.java     # Numerical gradient verification
│   ├── layers/
│   │   ├── Embedding.java           # Token embedding lookup
│   │   ├── PositionalEncoding.java  # Sinusoidal position encoding
│   │   ├── MultiHeadAttention.java  # Multi-head attention mechanism
│   │   ├── FeedForward.java         # Position-wise feed-forward network
│   │   ├── LayerNorm.java           # Layer normalization
│   │   └── Linear.java             # Linear projection (W*x + b)
│   ├── model/
│   │   ├── TransformerBlock.java    # Single transformer block
│   │   ├── Encoder.java             # Stack of encoder blocks
│   │   ├── Decoder.java             # Stack of decoder blocks
│   │   ├── EncoderDecoderModel.java # Full encoder-decoder transformer
│   │   └── DecoderOnlyModel.java    # GPT-style decoder-only transformer
│   ├── data/
│   │   ├── CharTokenizer.java       # Character-level tokenizer
│   │   ├── TextReader.java          # File reader for text corpus
│   │   ├── SequenceBatcher.java     # Batching sequences for training
│   │   └── ReversalTaskGenerator.java # Synthetic data for enc-dec task
│   ├── trainers/
│   │   └── AdamTrainer.java         # Adam optimizer
│   ├── CharGenerationTest.java      # Decoder-only demo
│   └── TranslationTest.java         # Encoder-decoder demo
```

### D. Model Size Estimates (for 1-hour training target)

| Parameter | Suggested Range | Notes |
|-----------|----------------|-------|
| Embedding dimension | 64-128 | Larger = more expressive but slower |
| Number of layers | 2-4 | More layers = deeper representations |
| Number of attention heads | 2-4 | Must evenly divide embedding dimension |
| Feed-forward inner dimension | 256-512 | Typically 4x embedding dimension |
| Context window | 64-128 characters | How much text the model sees at once |
| Batch size | 16-64 | Sequences per training step |
| Vocabulary size | ~65-95 | Determined by corpus (unique characters) |
| Total parameters | ~100K-500K | Tiny by modern standards, but enough to learn |

### E. Key Reference

- Vaswani, A. et al. (2017). "Attention Is All You Need." *arXiv:1706.03762*
- Karpathy, A. "nanoGPT" — minimal GPT implementation (conceptual reference, not code dependency)
- Kingma, D.P. & Ba, J. (2014). "Adam: A Method for Stochastic Optimization." *arXiv:1412.6980*
