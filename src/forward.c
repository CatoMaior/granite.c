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
        x[i] = m->token_embd[i][token_id] * EMBEDDING_SCALE;
    }

    // 2) passaggio nei layer (per ora non facciamo nulla, solo “echo”)
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
            sum += m->lm_head[v][i] * x_norm[i];
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
    matvec(gate, (const float *)L->w_gate, x_norm, D_MODEL, D_FF);
    matvec(up, (const float *)L->w_up, x_norm, D_MODEL, D_FF);

    // 3) hidden = SwiGLU(gate, up)
    swiglu(hidden, gate, up, D_FF);

    // 4) mlp_out = W_down * hidden
    matvec(mlp_out, (const float *)L->w_down, hidden, D_FF, D_MODEL);

    // 5) Residual: x += mlp_out
    for (int i = 0; i < D_MODEL; ++i) {
        x[i] += mlp_out[i];
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

    // 2) Q, K, V proiezioni lineari
    matvec(q, (const float *)L->w_q, x_norm, D_MODEL, D_MODEL);
    matvec(k_new, (const float *)L->w_k, x_norm, D_MODEL, N_KV_HEADS * HEAD_DIM);
    matvec(v_new, (const float *)L->w_v, x_norm, D_MODEL, N_KV_HEADS * HEAD_DIM);

    // 3) Applica RoPE a Q e K per questa posizione
    apply_rope_qk(q, k_new, pos);

    // 4) Scrivi K,V nella cache per questo token
    if (pos >= MAX_SEQ_LEN) {
        // per ora, se superiamo MAX_SEQ_LEN, tronchiamo (didattico)
        return;
    }
    for (int i = 0; i < N_KV_HEADS * HEAD_DIM; ++i) {
        cache->key_cache[layer_idx][pos][i] = k_new[i];
        cache->value_cache[layer_idx][pos][i] = v_new[i];
    }

    // 5) Attenzione multi-head
    float attn_out[D_MODEL]; // concatenazione delle 16 teste
    for (int i = 0; i < D_MODEL; ++i)
        attn_out[i] = 0.0f;

    int seq_len = pos + 1; // numero di token visti finora (0..pos)

    // scaling dell’attenzione (meta dice attention.scale = 1 / HEAD_DIM)
    const float scale = 1.0f / sqrtf((float)HEAD_DIM); // o 0.015625f

    // Loop sulle 16 teste Q
    for (int h = 0; h < N_HEADS; ++h) {

        int q_offset = h * HEAD_DIM; // in q[]
        // MQA: testa Q h usa la KV-head (h % N_KV_HEADS)
        int kv_head = h % N_KV_HEADS;
        int kv_offset = kv_head * HEAD_DIM;

        float scores[MAX_SEQ_LEN];

        // 6.a) Calcolo score per ogni posizione t (dot(Q_h, K_{kv_head, t}))
        for (int t = 0; t < seq_len; ++t) {
            float *k_t = &(cache->key_cache[layer_idx][t][kv_offset]);
            float s = 0.0f;
            for (int i = 0; i < HEAD_DIM; ++i) {
                s += q[q_offset + i] * k_t[i];
            }
            scores[t] = s * scale;
        }

        // 6.b) softmax sulle score
        softmax_inplace(scores, seq_len);

        // 6.c) combinazione pesata delle V
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

        // 6.d) scrivi questa testa nella posizione corretta di attn_out
        for (int i = 0; i < HEAD_DIM; ++i) {
            attn_out[q_offset + i] = head_out[i];
        }
    }

    // 7) Proiezione di output: attn_proj = attn_out * W_o
    float attn_proj[D_MODEL];
    matvec(attn_proj, (const float *)L->w_o, attn_out, D_MODEL, D_MODEL);

    // 8) Residual: x += attn_proj
    for (int i = 0; i < D_MODEL; ++i) {
        x[i] += attn_proj[i];
    }
}

void apply_rope_qk(float *q, float *k, int pos) {
    // caso limite
    if (ROPE_DIM > HEAD_DIM) return;

    // Precondizione: pos >= 0

    // 1) Applica RoPE a tutte le teste di Q (16 teste)
    for (int h = 0; h < N_HEADS; ++h) {
        float *qh = q + h * HEAD_DIM;

        // coppie (2i, 2i+1) fino a ROPE_DIM
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

    // 2) Applica RoPE alle teste K/V (4 teste)
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
