#include <math.h>
#include "forward.h"
#include "math_utils.h"
#include "model.h"

void forward_token(Model *m, int token_id, int pos, float *out_logits) {
    (void)pos; // non lo usiamo ancora

    float x[D_MODEL];

    // 1) embedding lookup
    for (int i = 0; i < D_MODEL; ++i) {
        x[i] = m->token_embd[i][token_id] * EMBEDDING_SCALE;
    }

    // 2) passaggio nei layer (per ora non facciamo nulla, solo “echo”)
    for (int l = 0; l < N_LAYERS; ++l) {
        (void)l;
        // TODO: qui metteremo attention + MLP
        // per ora lasciamo x invariato, o stampiamo qualcosa
    }

    // 3) output norm
    float x_norm[D_MODEL];
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
