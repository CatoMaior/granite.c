#include "model_init.h"
#include "dtype.h"
#include "model.h"
#include "weights_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void model_init_from_dir(Model *m, const char *base_dir) {
    // zero all fields
    memset(m, 0, sizeof(Model));

    model_set_meta(m);

    // 1) token embedding BF16: [D_MODEL, VOCAB_SIZE]
    load_bf16_tensor(
        base_dir,
        "token_embd.weight",
        &m->token_embd[0][0],
        (size_t)D_MODEL * (size_t)VOCAB_SIZE);

    // 2) output_norm F32: [D_MODEL]
    load_f32_tensor(
        base_dir,
        "output_norm.weight",
        m->output_norm,
        (size_t)D_MODEL);

    // 3) layers blk.0 ... blk.27
    for (int l = 0; l < N_LAYERS; ++l) {
        Layer *L = &m->layers[l];

        char name[64];

        // --- Norm F32 ---
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
        load_f32_tensor(
            base_dir,
            name,
            L->attn_norm,
            (size_t)D_MODEL);

        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
        load_f32_tensor(
            base_dir,
            name,
            L->ffn_norm,
            (size_t)D_MODEL);

        // --- Attention BF16 ---

        // attn_q.weight [1024,1024]
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
        load_bf16_tensor(
            base_dir,
            name,
            &L->w_q[0][0],
            (size_t)D_MODEL * (size_t)D_MODEL);

        // attn_k.weight [256,1024]
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
        load_bf16_tensor(
            base_dir,
            name,
            &L->w_k[0][0],
            (size_t)D_MODEL * (size_t)(N_KV_HEADS * HEAD_DIM));

        // attn_v.weight [256,1024]
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
        load_bf16_tensor(
            base_dir,
            name,
            &L->w_v[0][0],
            (size_t)D_MODEL * (size_t)(N_KV_HEADS * HEAD_DIM));

        // attn_output.weight [1024,1024]
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
        load_bf16_tensor(
            base_dir,
            name,
            &L->w_o[0][0],
            (size_t)D_MODEL * (size_t)D_MODEL);

        // --- MLP BF16 ---

        // ffn_gate.weight [2048,1024]
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", l);
        load_bf16_tensor(
            base_dir,
            name,
            &L->w_gate[0][0],
            (size_t)D_MODEL * (size_t)D_FF);

        // ffn_up.weight [1024,2048]
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", l);
        load_bf16_tensor(
            base_dir,
            name,
            &L->w_up[0][0],
            (size_t)D_MODEL * (size_t)D_FF);

        // ffn_down.weight [2048,1024]
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", l);
        load_bf16_tensor(
            base_dir,
            name,
            &L->w_down[0][0],
            (size_t)D_FF * (size_t)D_MODEL);
    }
}

void model_set_meta(Model *m) {
    m->vocab_size = VOCAB_SIZE;
    m->context_length = CONTEXT_LENGTH;
    m->n_layers = N_LAYERS;
    m->d_model = D_MODEL;
    m->n_heads = N_HEADS;
    m->n_kv_heads = N_KV_HEADS;
    m->head_dim = HEAD_DIM;
    m->d_ff = D_FF;
}

void model_init_rnd(Model *m) {
    memset(m, 0, sizeof(Model));
    model_set_meta(m);
    model_set_rnd_weights(m);
}

// Helper function to generate random float in given range
static inline bf16_t randbf16(float min, float max) {
    float result = min + ((float)rand() / (float)RAND_MAX) * (max - min);
    return f32_to_bf16(result);
}

void model_set_rnd_weights(Model *m) {
    // NOTE: matrix sizing might be transposed due to a previous implementation.
    // No need to change anything since weights are random.

    // Initialize random seed
    srand((unsigned int)42);

    // 1) Token embeddings [D_MODEL x VOCAB_SIZE]
    for (int i = 0; i < D_MODEL; ++i)
        for (int j = 0; j < VOCAB_SIZE; ++j)
            m->token_embd[j][i] = randbf16(-0.1f, 0.1f);

    // 2) Output normalization weights
    for (int i = 0; i < D_MODEL; ++i)
        m->output_norm[i] = 1.0f; // Initialize to 1.0 for layer norm

    // 4) Layer weights
    for (int l = 0; l < N_LAYERS; ++l) {
        Layer *layer = &m->layers[l];

        // Attention normalization
        for (int i = 0; i < D_MODEL; ++i)
            layer->attn_norm[i] = 1.0f;

        // Feed-forward normalization
        for (int i = 0; i < D_MODEL; ++i)
            layer->ffn_norm[i] = 1.0f;

        // Query projection [D_MODEL x D_MODEL]
        for (int i = 0; i < D_MODEL; ++i)
            for (int j = 0; j < D_MODEL; ++j)
                layer->w_q[i][j] = randbf16(-0.05f, 0.05f);

        // Key projection [D_MODEL x (N_KV_HEADS * HEAD_DIM)]
        for (int i = 0; i < D_MODEL; ++i)
            for (int j = 0; j < N_KV_HEADS * HEAD_DIM; ++j)
                layer->w_k[i][j] = randbf16(-0.05f, 0.05f);

        // Value projection [D_MODEL x (N_KV_HEADS * HEAD_DIM)]
        for (int i = 0; i < D_MODEL; ++i)
            for (int j = 0; j < N_KV_HEADS * HEAD_DIM; ++j)
                layer->w_v[i][j] = randbf16(-0.05f, 0.05f);

        // Output projection [D_MODEL x D_MODEL]
        for (int i = 0; i < D_MODEL; ++i)
            for (int j = 0; j < D_MODEL; ++j)
                layer->w_o[i][j] = randbf16(-0.05f, 0.05f);

        // Gate projection [D_MODEL x D_FF]
        for (int i = 0; i < D_MODEL; ++i)
            for (int j = 0; j < D_FF; ++j)
                layer->w_gate[j][i] = randbf16(-0.05f, 0.05f);

        // Up projection [D_MODEL x D_FF]
        for (int i = 0; i < D_MODEL; ++i)
            for (int j = 0; j < D_FF; ++j)
                layer->w_up[i][j] = randbf16(-0.05f, 0.05f);

        // Down projection [D_FF x D_MODEL]
        for (int i = 0; i < D_FF; ++i)
            for (int j = 0; j < D_MODEL; ++j)
                layer->w_down[i][j] = randbf16(-0.05f, 0.05f);
    }
}