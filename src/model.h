/**
 * @file model.h
 * @brief Granite model architecture definition and configuration
 *
 * This file defines the Granite transformer model architecture with its
 * hyperparameters, layer structures, and scaling factors.
 */

#pragma once
#include <stddef.h>
#include "dtype.h"

/** @brief Architecture identifier for Granite model */
#define ARCH_GRANITE 1

/** @brief Vocabulary size (number of tokens) */
#define VOCAB_SIZE 100352

/** @brief Maximum context length (sequence length) */
#define CONTEXT_LENGTH 32768

/** @brief Number of transformer layers */
#define N_LAYERS 28

/** @brief Model dimension (hidden size) */
#define D_MODEL 1024

/** @brief Number of attention heads */
#define N_HEADS 16

/** @brief Number of key-value heads (for grouped-query attention) */
#define N_KV_HEADS 4

/** @brief Dimension of each attention head */
#define HEAD_DIM 64

/** @brief Feed-forward network intermediate dimension */
#define D_FF 2048

/** @brief Embedding scaling factor */
#define EMBEDDING_SCALE 12.0f

/** @brief Residual connection scaling factor */
#define RESIDUAL_SCALE 0.2630000114f

/** @brief Logit scaling factor for output */
#define LOGIT_SCALE 4.0f

/** @brief Attention scaling factor for output */
#define ATTENTION_SCALE 0.015625f

/** @brief RMS normalization epsilon for numerical stability */
#define RMS_EPS 1e-5f

/** @brief Maximum length of input sequences */
#define MAX_SEQ_LEN 128 // original is 32768

/** @brief Rotary positional embedding dimension per head (RoPE) */
#define ROPE_DIM 64

/** @brief RoPE base frequency for rotary positional embeddings */
#define ROPE_BASE 10000000.0f

/**
 * @struct Layer
 * @brief Single transformer layer containing attention and MLP components
 *
 * This structure represents one layer of the transformer, including:
 * - Normalization weights for attention and feed-forward blocks
 * - Attention projection matrices (Q, K, V, O)
 * - MLP matrices for SwiGLU activation (gate, up, down)
 */
typedef struct {
    /** @brief Attention normalization weights [D_MODEL] */
    float attn_norm[D_MODEL]; // blk.X.attn_norm.weight [1024]

    /** @brief Feed-forward normalization weights [D_MODEL] */
    float ffn_norm[D_MODEL]; // blk.X.ffn_norm.weight  [1024]

    // Attention
    // Stored row-major: [in_dim][out_dim]

    /** @brief Query projection matrix [D_MODEL x D_MODEL] */
    bf16_t w_q[D_MODEL][D_MODEL];               // [1024, 1024]

    /** @brief Key projection matrix [D_MODEL x (N_KV_HEADS * HEAD_DIM)] */
    bf16_t w_k[N_KV_HEADS * HEAD_DIM][D_MODEL];

    /** @brief Value projection matrix [D_MODEL x (N_KV_HEADS * HEAD_DIM)] */
    bf16_t w_v[N_KV_HEADS * HEAD_DIM][D_MODEL];

    /** @brief Output projection matrix [D_MODEL x D_MODEL] */
    bf16_t w_o[D_MODEL][D_MODEL];               // [1024, 1024]

    // MLP (SwiGLU: gate/up/down)

    /** @brief Gate projection for SwiGLU activation [D_MODEL x D_FF] */
    bf16_t w_gate[D_FF][D_MODEL];

    /** @brief Up projection for SwiGLU activation [D_MODEL x D_FF] */
    bf16_t w_up[D_MODEL][D_FF];   // [1024, 2048]

    /** @brief Down projection for SwiGLU activation [D_FF x D_MODEL] */
    bf16_t w_down[D_FF][D_MODEL];
} Layer;

/**
 * @struct Model
 * @brief Complete Granite transformer model structure
 *
 * This structure contains all the model parameters including:
 * - Hyperparameters configuration
 * - Token embeddings
 * - Output normalization
 * - Language modeling head
 * - All transformer layers
 */
typedef struct {
    /** @brief Vocabulary size */
    int vocab_size;

    /** @brief Maximum context length */
    int context_length;

    /** @brief Number of layers */
    int n_layers;

    /** @brief Model dimension */
    int d_model;

    /** @brief Number of attention heads */
    int n_heads;

    /** @brief Number of key-value heads */
    int n_kv_heads;

    /** @brief Attention head dimension */
    int head_dim;

    /** @brief Feed-forward dimension */
    int d_ff;

    /** @brief Token embeddings [D_MODEL x VOCAB_SIZE] */
    // bf16_t token_embd[D_MODEL][VOCAB_SIZE];
    bf16_t token_embd[VOCAB_SIZE][D_MODEL];

    /** @brief Output normalization weights [D_MODEL] */
    float output_norm[D_MODEL];

    /** @brief Array of transformer layers [N_LAYERS] */
    Layer layers[N_LAYERS];

} Model;

/**
 * @struct KVCache
 * @brief Key-Value cache for efficient autoregressive generation
 *
 * This structure stores cached key and value projections from previous tokens
 * to avoid recomputing them during autoregressive text generation. The cache
 * is organized by layer and position for efficient access.
 */
typedef struct {
    /** @brief Cached key projections [N_LAYERS x MAX_SEQ_LEN x (N_KV_HEADS * HEAD_DIM)] */
    float key_cache[N_LAYERS][MAX_SEQ_LEN][N_KV_HEADS * HEAD_DIM];

    /** @brief Cached value projections [N_LAYERS x MAX_SEQ_LEN x (N_KV_HEADS * HEAD_DIM)] */
    float value_cache[N_LAYERS][MAX_SEQ_LEN][N_KV_HEADS * HEAD_DIM];
} KVCache;

