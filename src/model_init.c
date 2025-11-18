#include "model_init.h"

#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Helper function to generate random float in given range
static inline float randf(float min, float max) {
    return min + ((float)rand() / (float)RAND_MAX) * (max - min);
}

void model_init(Model *m) {
    memset(m, 0, sizeof(Model));

    m->vocab_size = VOCAB_SIZE;
    m->context_length = CONTEXT_LENGTH;
    m->n_layers = N_LAYERS;
    m->d_model = D_MODEL;
    m->n_heads = N_HEADS;
    m->n_kv_heads = N_KV_HEADS;
    m->head_dim = HEAD_DIM;
    m->d_ff = D_FF;

    model_load_weights(m);
}

void model_load_weights(Model *m) {
    // Initialize random seed
    srand((unsigned int)time(NULL));

    // 1) Token embeddings [D_MODEL x VOCAB_SIZE]
    for (int i = 0; i < D_MODEL; ++i) {
        for (int j = 0; j < VOCAB_SIZE; ++j) {
            m->token_embd[i][j] = randf(-0.1f, 0.1f);
        }
    }

    // 2) Output normalization weights
    for (int i = 0; i < D_MODEL; ++i) {
        m->output_norm[i] = 1.0f; // Initialize to 1.0 for layer norm
    }

    // 3) Language modeling head [VOCAB_SIZE x D_MODEL]
    for (int i = 0; i < VOCAB_SIZE; ++i) {
        for (int j = 0; j < D_MODEL; ++j) {
            m->lm_head[i][j] = randf(-0.1f, 0.1f);
        }
    }

    // 4) Layer weights
    for (int l = 0; l < N_LAYERS; ++l) {
        Layer *layer = &m->layers[l];

        // Attention normalization
        for (int i = 0; i < D_MODEL; ++i) {
            layer->attn_norm[i] = 1.0f;
        }

        // Feed-forward normalization
        for (int i = 0; i < D_MODEL; ++i) {
            layer->ffn_norm[i] = 1.0f;
        }

        // Query projection [D_MODEL x D_MODEL]
        for (int i = 0; i < D_MODEL; ++i) {
            for (int j = 0; j < D_MODEL; ++j) {
                layer->w_q[i][j] = randf(-0.05f, 0.05f);
            }
        }

        // Key projection [D_MODEL x (N_KV_HEADS * HEAD_DIM)]
        for (int i = 0; i < D_MODEL; ++i) {
            for (int j = 0; j < N_KV_HEADS * HEAD_DIM; ++j) {
                layer->w_k[i][j] = randf(-0.05f, 0.05f);
            }
        }

        // Value projection [D_MODEL x (N_KV_HEADS * HEAD_DIM)]
        for (int i = 0; i < D_MODEL; ++i) {
            for (int j = 0; j < N_KV_HEADS * HEAD_DIM; ++j) {
                layer->w_v[i][j] = randf(-0.05f, 0.05f);
            }
        }

        // Output projection [D_MODEL x D_MODEL]
        for (int i = 0; i < D_MODEL; ++i) {
            for (int j = 0; j < D_MODEL; ++j) {
                layer->w_o[i][j] = randf(-0.05f, 0.05f);
            }
        }

        // Gate projection [D_MODEL x D_FF]
        for (int i = 0; i < D_MODEL; ++i) {
            for (int j = 0; j < D_FF; ++j) {
                layer->w_gate[i][j] = randf(-0.05f, 0.05f);
            }
        }

        // Up projection [D_MODEL x D_FF]
        for (int i = 0; i < D_MODEL; ++i) {
            for (int j = 0; j < D_FF; ++j) {
                layer->w_up[i][j] = randf(-0.05f, 0.05f);
            }
        }

        // Down projection [D_FF x D_MODEL]
        for (int i = 0; i < D_FF; ++i) {
            for (int j = 0; j < D_MODEL; ++j) {
                layer->w_down[i][j] = randf(-0.05f, 0.05f);
            }
        }
    }
}

