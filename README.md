# Granite.c

A minimal, educational implementation of the IBM Granite 4.0 350M transformer model in pure C. This project demonstrates the core concepts of large language model inference without relying on external deep learning frameworks.

> **⚠️ Educational Purpose Only**
> 
> This project is designed for learning and understanding transformer architectures. It prioritizes code clarity and readability over performance optimization. Do not expect production-level speed or efficiency.

## Overview

Granite.c is a from-scratch implementation of a decoder-only transformer model that can run inference on the IBM Granite 4.0 350M parameter model. The codebase demonstrates:

- **Transformer Architecture**: Multi-head grouped-query attention, SwiGLU activations, RoPE positional embeddings
- **Custom Data Types**: BFloat16 (BF16) weight storage with runtime conversion to FP32
- **Memory Management**: Manual memory allocation and KV-cache implementation
- **Tokenization**: Basic BPE-style tokenizer with vocabulary lookup
- **Autoregressive Generation**: Token-by-token text generation with streaming output

## Architecture Details

The implementation includes the following components:

### Model Configuration (Granite 4.0 350M)
- **Vocabulary Size**: 100,352 tokens
- **Context Length**: 32,768 tokens
- **Model Dimension**: 1,024
- **Number of Layers**: 28
- **Attention Heads**: 16 (with 4 KV heads for grouped-query attention)
- **Feed-Forward Dimension**: 2,048
- **Positional Encoding**: RoPE (Rotary Position Embedding)

### Key Features
- **Grouped-Query Attention (GQA)**: Reduces KV-cache memory by sharing key/value heads across query heads
- **SwiGLU Activation**: Gated activation function in the feed-forward network
- **RMS Normalization**: Efficient normalization alternative to LayerNorm
- **BFloat16 Weights**: Reduces model size while maintaining float32 dynamic range
- **Streaming Output**: Real-time token generation with terminal output

## Project Structure

```
granite.c/
├── src/
│   ├── main.c              # Entry point and generation loop
│   ├── model.h             # Model architecture and hyperparameters
│   ├── model_init.c        # Model initialization and weight loading
│   ├── forward.c           # Forward pass implementation
│   ├── tokenizer.c         # Tokenization and detokenization
│   ├── weights_loader.c    # Binary weight file loading
│   ├── dtype.h             # BFloat16 type and conversions
│   ├── math_utils.h        # Matrix operations and normalization
│   ├── activations.h       # Activation functions (SiLU, SwiGLU)
│   └── utils.h             # Utility macros
├── granite-4.0-350m-BF16/  # Model weights directory
│   ├── model.gguf          # Original GGUF format model
│   ├── vocab.txt           # Vocabulary file
│   ├── weights_index.json  # Weight metadata
│   └── weights/            # Extracted binary weight files
├── extract_weights.py      # GGUF to binary weight converter
├── export_vocab.py         # Vocabulary extraction script
├── Makefile                # Build configuration
└── README.md               # This file
```

## Prerequisites

- **C Compiler**: GCC with OpenMP support
- **Python 3.8+**: For weight extraction scripts
- **Python Libraries**: 
  - `numpy`
  - `gguf` (for GGUF file reading)
  - `transformers` (for vocabulary extraction)

## Getting Started

### 1. Download the Model

Download the IBM Granite 4.0 350M model and place it in the project directory:

```bash
mkdir -p granite-4.0-350m-BF16
wget https://huggingface.co/ibm-granite/granite-4.0-350m-GGUF/resolve/main/granite-4.0-350m-bf16.gguf?download=true -O granite-4.0-350m-BF16/model.gguf
```

### 2. Extract Weights and Vocabulary

Run the Python scripts to convert the model weights from GGUF format to binary files:

```bash
# Activate virtual environment (if using one)
python3 -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate   # On Windows

# Install dependencies
pip install numpy gguf transformers

# Extract weights from GGUF file
python3 extract_weights.py

# Export vocabulary
python3 export_vocab.py
```

This will create the `granite-4.0-350m-BF16/weights/` directory with individual binary files for each tensor and a `vocab.txt` file.

### 3. Build the Project

Compile the C source code using the provided Makefile:

```bash
make
```

This will create the executable at `build/granite-c`.

### 4. Run Inference

Execute the compiled binary:

```bash
./build/granite-c
```

The program will generate text starting from the initial token "Hello" and display the output in real-time.

## Code Walkthrough

### Forward Pass (`src/forward.c`)

The forward pass consists of:

1. **Embedding Lookup**: Convert token ID to embedding vector with scaling
2. **Layer Processing**: 28 transformer layers, each with:
   - RMS normalization
   - Multi-head grouped-query attention with RoPE
   - Residual connection
   - RMS normalization
   - SwiGLU feed-forward network
   - Residual connection
3. **Output Normalization**: Final RMS norm
4. **Logit Computation**: Matrix multiplication with embedding weights (tied weights)

### Attention Mechanism

The implementation uses Grouped-Query Attention (GQA):
- 16 query heads
- 4 key/value heads (each KV head is shared by 4 query heads)
- Rotary Position Embeddings (RoPE) applied to queries and keys
- Attention scores scaled by `1/sqrt(head_dim) = 0.015625`

### KV-Cache

To avoid recomputing attention for previous tokens during autoregressive generation, the model maintains a key-value cache:
- Stores key and value projections for each layer and position
- Dimensions: `[N_LAYERS][MAX_SEQ_LEN][N_KV_HEADS * HEAD_DIM]`

### Tokenization

The tokenizer implements:
- Vocabulary lookup from `vocab.txt`
- Special token handling (spaces as `Ġ`, newlines as `Ċ`)
- Detokenization with proper spacing and formatting

## Performance Notes

This implementation is **not optimized for speed** with basic OpenMP parallelization only.

**Expected Performance**: Slow. This is intended for educational purposes to understand how transformers work at a low level.

## Customization

### Changing the Model

To use a different model or variant:
1. Update the constants in `src/model.h` (vocabulary size, layers, dimensions, etc.)
2. Modify `BASE_DIR` in `src/main.c` to point to your model directory
3. Ensure weight extraction scripts handle your model format

### Adjusting Generation

Modify in `src/main.c`:
- `NUM_TOKENS`: Number of tokens to generate
- `MAX_SEQ_LEN`: Maximum sequence length (in `src/model.h`)
- Initial prompt tokens

### Sampling Strategy

Currently uses greedy decoding (argmax). To implement other sampling methods:
- Modify the sampling logic in `src/main.c` after `forward_token()`
- Consider temperature scaling, top-k, top-p, or other techniques

## Build Options

The Makefile uses these compiler flags:

```makefile
CFLAGS = -Wall -Wextra -g -O0 -lm -fopenmp -ffast-math
```

- `-g`: Debug symbols
- `-O0`: No optimization (for debugging; use `-O3` for better performance)
- `-fopenmp`: OpenMP support for parallelization
- `-ffast-math`: Fast floating-point math

## Common Issues

### Segmentation Fault
- Check that `MAX_SEQ_LEN` is not exceeded
- Ensure model weights are correctly loaded
- Verify memory allocation for `Model` and `KVCache`

### Incorrect Output
- Verify vocabulary file matches the model
- Check that weight files are correctly extracted
- Ensure BF16 to FP32 conversion is working

### Slow Performance
- This is expected! The code prioritizes clarity over speed
- For faster inference, consider using optimized frameworks (llama.cpp, GGML, etc.)

## Learning Resources

This implementation demonstrates concepts from:

- **"Attention Is All You Need"** (Vaswani et al., 2017) - Original Transformer paper
- **"RoFormer: Enhanced Transformer with Rotary Position Embedding"** (Su et al., 2021)
- **"GQA: Training Generalized Multi-Query Transformer"** (Ainslie et al., 2023)
- **"GLU Variants Improve Transformer"** (Shazeer, 2020) - SwiGLU activation

## License

This project is for educational purposes. Please refer to IBM Granite's license for model usage terms.