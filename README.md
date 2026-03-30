# JavaLLM

A from-scratch Transformer implementation in pure Java. Every component — autograd engine, attention layers, training loop, tokenizers — is hand-written with no external machine learning libraries. The goal is to make Transformers understandable by building every piece yourself.

JavaLLM implements both a **decoder-only model** (GPT-style, for text generation) and an **encoder-decoder model** (for sequence-to-sequence tasks like string reversal). It includes two tokenizers — character-level and word-level — to demonstrate that the Transformer architecture is agnostic to how text is split into tokens.

## Quick Start

**Prerequisites:** Java 21+ (`java -version` to check)

```bash
cd JavaLLM

# Build (downloads Maven automatically if not installed)
./mvnw package

# Download training data
bash data/download-shakespeare.sh

# Train a character-level model on Shakespeare (500 steps, ~5 minutes)
java -jar target/javallm.jar train --data data/shakespeare.txt

# Generate text from the trained model
java -jar target/javallm.jar generate --model out.model --data data/shakespeare.txt --prompt "To be or not"

# Or enter interactive mode (no --prompt)
java -jar target/javallm.jar generate --model out.model --data data/shakespeare.txt
```

### Running the Demo Programs Directly

The original demo programs are still available:

```bash
# Shakespeare text generation (trains + interactive mode)
java -cp target/classes org.ea.javallm.CharGenerationTest

# String reversal (encoder-decoder)
java -cp target/classes org.ea.javallm.TranslationTest
```

### Running Tests

```bash
./mvnw test
# Runs 131 JUnit 5 tests covering all components
```

## CLI Reference

### `train` — Train a model

```
java -jar javallm.jar train --data <path> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data <path>` | *(required)* | Training data text file |
| `--model <path>` | `out.model` | Where to save the trained model |
| `--tokenizer <type>` | `char` | Tokenizer: `char` or `word` |
| `--embed-dim <n>` | `64` | Embedding dimension |
| `--layers <n>` | `2` | Number of Transformer layers |
| `--heads <n>` | `2` | Number of attention heads |
| `--ffn-dim <n>` | `256` | Feed-forward inner dimension |
| `--context-len <n>` | `32` | Training context length |
| `--batch-size <n>` | `8` | Training batch size |
| `--steps <n>` | `500` | Number of training steps |
| `--learning-rate <r>` | `3e-4` | Adam optimizer learning rate |

### `generate` — Generate text

```
java -jar javallm.jar generate --model <path> --data <path> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model <path>` | *(required)* | Path to a saved model |
| `--data <path>` | *(required)* | Training data (to rebuild tokenizer vocabulary) |
| `--tokenizer <type>` | `char` | Must match the tokenizer used during training |
| `--prompt <text>` | *(interactive)* | Text to continue; omit for interactive mode |
| `--temperature <t>` | `0.8` | Sampling temperature (lower = more deterministic) |
| `--length <n>` | `100` | Maximum tokens to generate |

Model architecture flags (`--embed-dim`, `--layers`, `--heads`, `--ffn-dim`, `--context-len`) must match the values used during training.

## Tokenizers: Characters vs. Words

JavaLLM includes two tokenizers to illustrate that a "token" can be any unit of text:

**CharTokenizer** splits text into individual characters. Given `"hello"`, it produces 5 tokens: `h`, `e`, `l`, `l`, `o`. This results in a small vocabulary (65 characters for Shakespeare) but long sequences.

**WordTokenizer** splits text on whitespace boundaries. Given `"to be or not to be"`, it produces 6 tokens: `to`, `be`, `or`, `not`, `to`, `be`. This results in shorter sequences but a much larger vocabulary (~25,000 unique words for Shakespeare).

The same Transformer model works with either tokenizer — the only thing that changes is the vocabulary size (and therefore the embedding matrix size). This demonstrates that the attention mechanism operates on abstract token IDs, not on characters or words directly.

```bash
# Train with character tokenizer (default)
java -jar javallm.jar train --data data/shakespeare.txt --tokenizer char

# Train with word tokenizer
java -jar javallm.jar train --data data/shakespeare.txt --tokenizer word
```

## Architecture

The project is organized into six packages that mirror the logical layers of a Transformer:

```
org.ea.javallm
├── autograd/              Automatic differentiation engine
│   ├── Tensor             N-dimensional tensor with gradient tracking
│   ├── Operation          Base class for differentiable operations
│   ├── GradientChecker    Numerical gradient validation
│   └── ops/               Operation implementations
│       ├── MatMul          Matrix multiplication (2D and batched 3D)
│       ├── Add             Element-wise addition with broadcasting
│       ├── Multiply        Hadamard (element-wise) product
│       ├── Scale           Scalar multiplication
│       ├── Softmax         Numerically stable softmax
│       ├── ReLU            Rectified linear unit activation
│       ├── Transpose       Dimension swapping
│       ├── Reshape         Shape change without data copy
│       ├── LayerNormOp     Layer normalization forward/backward
│       ├── CrossEntropy    Cross-entropy loss from logits
│       ├── EmbeddingLookup Row selection by integer indices
│       └── Mask            Causal attention masking
│
├── layers/                Transformer building blocks
│   ├── Linear             Fully connected layer (y = xW^T + b)
│   ├── Embedding          Token-to-vector lookup table
│   ├── PositionalEncoding Sinusoidal position signals
│   ├── LayerNorm          Layer normalization with learnable parameters
│   ├── FeedForward        Two-layer MLP with ReLU
│   └── MultiHeadAttention Scaled dot-product attention with multiple heads
│
├── model/                 Complete model architectures
│   ├── TransformerBlock   Single Transformer block (attention + FFN + residuals)
│   ├── DecoderOnlyModel   GPT-style model for text generation
│   └── EncoderDecoderModel Full encoder-decoder for seq-to-seq tasks
│
├── data/                  Tokenization and data loading
│   ├── Tokenizer          Common interface for all tokenizers
│   ├── CharTokenizer      Character-level tokenizer
│   ├── WordTokenizer      Word-level tokenizer
│   ├── TextReader         File reader with train/validation split
│   ├── SequenceBatcher    Batch producer for decoder-only training
│   └── ReversalTaskGenerator Synthetic data for encoder-decoder training
│
├── trainers/              Optimization and persistence
│   ├── AdamOptimizer      Adam optimizer with bias correction
│   └── ModelIO            Plain-text model save/load
│
├── Main                   CLI entry point (train / generate)
├── TextGenerator          Shared autoregressive generation logic
├── CharGenerationTest     Shakespeare text generation demo
└── TranslationTest        String reversal demo
```

### Key Design Patterns

- **Dynamic computation graph**: Operations build a graph at runtime (like PyTorch). `tensor.backward()` walks the graph in reverse to compute gradients.
- **Pre-norm residuals**: LayerNorm is applied *before* each sublayer, which is more stable than the original post-norm design.
- **Weight tying**: In the decoder-only model, the token embedding matrix is reused as the output projection — reducing parameter count and improving training.

## Example Output

Training on Shakespeare with default settings (500 steps, ~5 min on CPU):

```
$ java -jar javallm.jar train --data data/shakespeare.txt --steps 100

=== JavaLLM Training ===

Training text: 1003854 characters
Tokenizer: char
Vocabulary size: 65
Token count: 1003854
Model parameters: 104256

Training for 100 steps...

Step 2/100  loss=4.2540
Step 10/100  loss=3.6565
Step 20/100  loss=3.3616

--- Sample at step 20 ---
The rh r;hsraea vW sa
V t:LomwrttwtenFoo   arshBJoo.Ta
--- End sample ---

Step 40/100  loss=3.2699
Step 60/100  loss=3.2134
Step 80/100  loss=3.2918
Step 100/100  loss=3.2057

--- Sample at step 100 ---
The  nfhsd:ho ss a, c  eat   tta  h
tttsml   e
wtoieko
--- End sample ---

Saving model to out.model...
Model saved.
```

With longer training (500 steps), the model begins producing recognizable English words and Shakespeare-like patterns. The loss decreases from ~4.2 (random) toward ~2.5.

## Project Structure

```
JavaLLM/
├── pom.xml                    Maven build configuration
├── mvnw / mvnw.cmd            Maven wrapper (no pre-install needed)
├── data/
│   ├── download-shakespeare.sh  Script to download training data
│   └── shakespeare.txt          Tiny Shakespeare corpus (~1MB)
├── src/
│   ├── main/java/org/ea/javallm/   Source code (35 Java files)
│   └── test/java/org/ea/javallm/   Tests (19 test files, 131 tests)
└── target/
    └── javallm.jar              Built fat JAR (after mvn package)
```

## License

MIT License. Copyright 2026 Daniel Persson.
