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
 * 1. RMS normalization of input
 * 2. Parallel gate and up projections
 * 3. SwiGLU activation (gated activation)
 * 4. Down projection back to model dimension
 * 5. Residual connection (adds result to input x)
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
 * 1. RMS normalization of input
 * 2. Compute Q, K, V projections
 * 3. Apply RoPE (Rotary Position Embedding) to Q and K
 * 4. Store K, V in cache for current position
 * 5. Compute attention scores with all cached keys
 * 6. Apply softmax to get attention weights
 * 7. Weighted sum of values
 * 8. Output projection
 * 9. Residual connection (adds result to input x)
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
 * 1. Compute rotation coefficients (cos, sin) for this position and feature indices.
 * 2. For each pair (2*i, 2*i+1) perform the complex rotation:
 *    q_even' = q_even * cos - q_odd * sin
 *    q_odd'  = q_even * sin + q_odd * cos
 *    (same for k)
 * 3. Store rotated components back into q and k.
 *
 * @param q Pointer to the query vector for a single token (modified in-place).
 * @param k Pointer to the key vector for a single token (modified in-place).
 * @param pos Token position (non-negative).
 */
void apply_rope_qk(float *q, float *k, int pos);