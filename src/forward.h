/**
 * @file forward.h
 * @brief Forward pass declaration for the Granite model
 *
 * This file contains the function declaration for performing a forward pass
 * through the Granite transformer model.
 */

#pragma once
#include "model.h"

/**
 * @brief Perform a forward pass for a single token
 *
 * This function takes a token ID and runs it through the entire model
 * to produce output logits for all vocabulary tokens. It processes the token
 * through all layers using cached key-value pairs for efficient autoregressive generation.
 *
 * Steps:
 * 1. Embedding lookup
 * 2. Forward through layers (attention and MLP)
 * 3. Output normalization
 * 4. Compute logits via lm_head projection
 *
 * @param m Pointer to the Model structure containing all weights
 * @param cache Pointer to the KVCache for storing/retrieving key-value pairs across layers
 * @param token_id Input token ID (0 to VOCAB_SIZE-1)
 * @param pos Position of the token in the sequence (0 to MAX_SEQ_LEN-1)
 * @param out_logits Output array to store logits [VOCAB_SIZE]
 */
void forward_token(Model *m, KVCache *cache, int token_id, int pos, float *out_logits);

/**
 * @brief Forward pass through a single layer's MLP (feed-forward) block
 *
 * Performs the MLP transformation for one transformer layer:
 * 1. RMS normalization of input (x_norm = RMSNorm(x) with ffn_norm)
 * 2. Gate and up projections (gate = W_gate * x_norm, up = W_up * x_norm)
 * 3. SwiGLU activation (hidden = SwiGLU(gate, up))
 * 4. Down projection (mlp_out = W_down * hidden)
 * 5. Residual connection (x += mlp_out)
 *
 * @param m Pointer to the Model structure containing all weights
 * @param layer_idx Index of the layer (0 to N_LAYERS-1)
 * @param x Input/output vector of size D_MODEL (modified in-place with residual)
 */
void forward_layer_mlp(Model *m, int layer_idx, float *x);

/**
 * @brief Forward pass through a single layer's attention block
 *
 * Performs the multi-head grouped-query attention (GQA) for one transformer layer:
 * 1. RMS normalization
 * 2. Q, K, V linear projections
 * 3. Apply RoPE to Q and K for this position
 * 4. Write K, V to cache for this token
 * 5. Multi-head attention (compute scores, softmax, weighted combination)
 * 6. (Sub-steps within 5: compute scores for each position, softmax, weighted combination of V)
 * 7. Output projection (attn_proj = attn_out * W_o)
 * 8. Residual connection (x += attn_proj)
 *
 * @param m Pointer to the Model structure containing all weights
 * @param cache Pointer to the KVCache for storing/retrieving key-value pairs
 * @param layer_idx Index of the layer (0 to N_LAYERS-1)
 * @param pos Current position in the sequence (0 to MAX_SEQ_LEN-1)
 * @param x Input/output vector of size D_MODEL (modified in-place with residual)
 */
void forward_layer_attn(Model *m, KVCache *cache, int layer_idx, int pos, float *x);

/**
 * @brief Apply rotary positional embeddings (RoPE) to query and key vectors for a given token position.
 *
 * Applies the RoPE rotation in-place to the provided per-head query and key vectors for the specified
 * token position. The feature dimension is treated as interleaved (even/odd) pairs and must be even.
 *
 * Steps:
 * 1. Apply RoPE to Q heads (16 heads)
 *    - For each pair (2i, 2i+1) up to ROPE_DIM, compute rotation coefficients (cos, sin)
 *    - Perform complex rotation: q_even' = q_even * cos - q_odd * sin, q_odd' = q_even * sin + q_odd * cos
 * 2. Apply RoPE to K/V heads (4 heads)
 *    - Same rotation process as for Q heads
 *
 * @param q Pointer to the query vector for a single token (modified in-place).
 * @param k Pointer to the key vector for a single token (modified in-place).
 * @param pos Token position (non-negative).
 */
void apply_rope_qk(float *q, float *k, int pos);

/**
 * @brief Initialize the global RoPE cache with precomputed cos/sin values
 *
 * This function must be called once at program startup before any forward passes.
 * It precomputes all cosine and sine values for rotary positional embeddings
 * to avoid expensive trigonometric operations during inference.
 */
void rope_cache_init(void);