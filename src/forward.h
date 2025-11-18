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
 * to produce output logits for all vocabulary tokens.
 *
 * @param m Pointer to the Model structure containing all weights
 * @param token_id Input token ID (0 to VOCAB_SIZE-1)
 * @param pos Position of the token in the sequence (currently unused)
 * @param out_logits Output array to store logits [VOCAB_SIZE]
 */
void forward_token(Model *m, int token_id, int pos, float *out_logits);

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