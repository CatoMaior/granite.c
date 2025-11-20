#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "forward.h"
#include "model.h"
#include "utils.h"
#include "math_utils.h"
#include "model_init.h"
#include "weights_loader.h"

#define NUM_TOKENS 1
#define BASE_DIR "granite-4.0-350m-BF16"

int main(void) {
    // On heap because model is too large for stack
    Model *m = malloc(sizeof(Model));
    CHECK_PTR(m, "malloc model");

    print_test_tensor(BASE_DIR);

    model_init_from_dir(m, BASE_DIR);

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
