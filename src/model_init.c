#include "model_init.h"

#include "model.h"
#include <string.h>

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

    // Esempio: inizializza l’embedding del token 0 a una base canonica
    for (int i = 0; i < D_MODEL; ++i) {
        m->token_embd[i][0] = (i == 0) ? 1.0f : 0.0f;
    }
}