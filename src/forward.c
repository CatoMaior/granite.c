#include "forward.h"
#include "activations.h"
#include "math_utils.h"
#include "model.h"
#include <math.h>

void forward_token(Model *m, KVCache *cache, int token_id, int pos, float *out_logits) {

    float x[D_MODEL];
    float x_norm[D_MODEL];

    // 1) embedding lookup
    for (int i = 0; i < D_MODEL; ++i) {
        bf16_t w_bf16 = m->token_embd[token_id][i];
        float w = bf16_to_f32(w_bf16);
        x[i] = w * EMBEDDING_SCALE;
    }

    // 2) forward through layers
    for (int l = 0; l < N_LAYERS; ++l) {
        forward_layer_attn(m, cache, l, pos, x);
        forward_layer_mlp(m, l, x);
    }

    // 3) output norm
    rms_norm(x_norm, x, m->output_norm, D_MODEL, RMS_EPS);

    // 4) logits = x_norm * lm_head
    for (int v = 0; v < VOCAB_SIZE; ++v) {
        float sum = 0.0f;
        for (int i = 0; i < D_MODEL; ++i) {
            bf16_t w_bf16 = m->token_embd[v][i];
            float w = bf16_to_f32(w_bf16);
            sum += w * x_norm[i];
        }
        out_logits[v] = sum * LOGIT_SCALE;
    }
}

void forward_layer_mlp(Model *m, int layer_idx, float *x) {
    Layer *L = &(m->layers[layer_idx]);

    float x_norm[D_MODEL];
    float gate[D_FF];
    float up[D_FF];
    float hidden[D_FF];
    float mlp_out[D_MODEL];

    // 1) RMSNorm: x_norm = RMSNorm(x) con ffn_norm
    rms_norm(x_norm, x, L->ffn_norm, D_MODEL, RMS_EPS);

    // 2) gate = W_gate * x_norm, up = W_up * x_norm
    matvec_bf16(gate, (const bf16_t *)L->w_gate, x_norm, D_MODEL, D_FF);
    matvec_bf16(up, (const bf16_t *)L->w_up, x_norm, D_MODEL, D_FF);

    // 3) hidden = SwiGLU(gate, up)
    swiglu(hidden, gate, up, D_FF);

    // 4) mlp_out = W_down * hidden
    matvec_bf16(mlp_out, (const bf16_t *)L->w_down, hidden, D_FF, D_MODEL);

    // 5) Residual: x += mlp_out
    for (int i = 0; i < D_MODEL; ++i) {
        x[i] += mlp_out[i] * RESIDUAL_SCALE;
    }
}

void forward_layer_attn(Model *m, KVCache *cache, int layer_idx, int pos, float *x) {
    Layer *L = &(m->layers[layer_idx]);

    float x_norm[D_MODEL];
    float q[D_MODEL];                   // 16 * 64 = 1024
    float k_new[N_KV_HEADS * HEAD_DIM]; // 4 * 64 = 256
    float v_new[N_KV_HEADS * HEAD_DIM];

    // 1) RMSNorm
    rms_norm(x_norm, x, L->attn_norm, D_MODEL, RMS_EPS);

    // 2) Q, K, V linear projections
    matvec_bf16(q, (const bf16_t *)L->w_q, x_norm, D_MODEL, D_MODEL);
    matvec_bf16(k_new, (const bf16_t *)L->w_k, x_norm, D_MODEL, N_KV_HEADS * HEAD_DIM);
    matvec_bf16(v_new, (const bf16_t *)L->w_v, x_norm, D_MODEL, N_KV_HEADS * HEAD_DIM);

    // 3) Apply RoPE to Q and K for this position
    apply_rope_qk(q, k_new, pos);

    // 4) Write K,V to cache for this token
    if (pos >= MAX_SEQ_LEN) {
        // if we exceed MAX_SEQ_LEN, we truncate (didactic)
        return;
    }
    for (int i = 0; i < N_KV_HEADS * HEAD_DIM; ++i) {
        cache->key_cache[layer_idx][pos][i] = k_new[i];
        cache->value_cache[layer_idx][pos][i] = v_new[i];
    }

    // 5) Multi-head attention
    float attn_out[D_MODEL]; // concatenation of the 16 heads
    for (int i = 0; i < D_MODEL; ++i)
        attn_out[i] = 0.0f;

    int seq_len = pos + 1; // number of tokens seen so far (0..pos)

    // Loop over the 16 heads
    for (int h = 0; h < N_HEADS; ++h) {

        int q_offset = h * HEAD_DIM; // inside q[]
        int n_rep = N_HEADS / N_KV_HEADS; // 16 / 4 = 4
        int kv_head = h / n_rep;          // Q0,Q1,Q2,Q3 -> K0; Q4... -> K1
        int kv_offset = kv_head * HEAD_DIM;

        float scores[MAX_SEQ_LEN];

        // 6.a) Compute score for each position t (dot(Q_h, K_{kv_head, t}))
        for (int t = 0; t < seq_len; ++t) {
            float *k_t = &(cache->key_cache[layer_idx][t][kv_offset]);
            float s = 0.0f;
            for (int i = 0; i < HEAD_DIM; ++i) {
                s += q[q_offset + i] * k_t[i];
            }
            scores[t] = s * ATTENTION_SCALE;
        }

        // 6.b) softmax on scores
        softmax_inplace(scores, seq_len);

        // 6.c) weighted combination of V
        float head_out[HEAD_DIM];
        for (int i = 0; i < HEAD_DIM; ++i)
            head_out[i] = 0.0f;

        for (int t = 0; t < seq_len; ++t) {
            float weight = scores[t];
            float *v_t = &(cache->value_cache[layer_idx][t][kv_offset]);
            for (int i = 0; i < HEAD_DIM; ++i) {
                head_out[i] += weight * v_t[i];
            }
        }

        // 6.d) Write this head to the correct position in attn_out
        for (int i = 0; i < HEAD_DIM; ++i) {
            attn_out[q_offset + i] = head_out[i];
        }
    }

    // 7) Output projection: attn_proj = attn_out * W_o
    float attn_proj[D_MODEL];
    matvec_bf16(attn_proj, (const bf16_t *)L->w_o, attn_out, D_MODEL, D_MODEL);

    // 8) Residual: x += attn_proj
    for (int i = 0; i < D_MODEL; ++i) {
        x[i] += attn_proj[i] * RESIDUAL_SCALE;
    }
}

void apply_rope_qk(float *q, float *k, int pos) {
    // Limit case
    if (ROPE_DIM > HEAD_DIM) return;

    // Precondition: pos >= 0

    // 1) Apply RoPE to Q heads (16 heads)
    for (int h = 0; h < N_HEADS; ++h) {
        float *qh = q + h * HEAD_DIM;

        // pairs (2i, 2i+1) up to ROPE_DIM
        for (int i = 0; i < ROPE_DIM; i += 2) {
            int pair_index = i / 2;

            // freq_i = base^(-2 * pair_index / ROPE_DIM)
            float exponent = -2.0f * (float)pair_index / (float)ROPE_DIM;
            float freq = powf(ROPE_BASE, exponent);

            float angle = (float)pos * freq;
            float c = cosf(angle);
            float s = sinf(angle);

            float x0 = qh[i];
            float x1 = qh[i + 1];

            qh[i] = x0 * c - x1 * s;
            qh[i + 1] = x0 * s + x1 * c;
        }
    }

    // 2) Apply RoPE to K/V heads (4 heads)
    for (int h = 0; h < N_KV_HEADS; ++h) {
        float *kh = k + h * HEAD_DIM;

        for (int i = 0; i < ROPE_DIM; i += 2) {
            int pair_index = i / 2;

            float exponent = -2.0f * (float)pair_index / (float)ROPE_DIM;
            float freq = powf(ROPE_BASE, exponent);

            float angle = (float)pos * freq;
            float c = cosf(angle);
            float s = sinf(angle);

            float x0 = kh[i];
            float x1 = kh[i + 1];

            kh[i] = x0 * c - x1 * s;
            kh[i + 1] = x0 * s + x1 * c;
        }
    }
}
