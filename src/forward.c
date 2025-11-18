#include "forward.h"
#include "activations.h"
#include "math_utils.h"
#include "model.h"
#include <math.h>

void forward_token(Model *m, int token_id, int pos, float *out_logits) {
    (void)pos; // non lo usiamo ancora

    float x[D_MODEL];
    float x_norm[D_MODEL];

    // 1) embedding lookup
    for (int i = 0; i < D_MODEL; ++i) {
        x[i] = m->token_embd[i][token_id] * EMBEDDING_SCALE;
    }

    // 2) passaggio nei layer (per ora non facciamo nulla, solo “echo”)
    for (int l = 0; l < N_LAYERS; ++l) {
        (void)l;
        // TODO: qui metteremo attention + MLP
        // per ora lasciamo x invariato, o stampiamo qualcosa
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
