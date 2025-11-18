#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "forward.h"
#include "model.h"
#include "utils.h"
#include "model_init.h"

#define NUM_TOKENS 10

int main(void) {
    // On heap because model is too large for stack
    Model *m = malloc(sizeof(Model));
    CHECK_PTR(m, "malloc model");
    model_init(m);

    KVCache *cache = malloc(sizeof(KVCache));
    CHECK_PTR(cache, "malloc KVCache");
    memset(cache, 0, sizeof(KVCache));

    float *logits = malloc(sizeof(float) * VOCAB_SIZE);
    CHECK_PTR(logits, "malloc logits");

    for (int i = 0; i < NUM_TOKENS; i++) {
        forward_token(m, cache, i, 0, logits);
    }

    printf("logit[%d] = %f\n", 0, logits[0]);

    free(logits);
    free(cache);
    free(m);
    return 0;
}
