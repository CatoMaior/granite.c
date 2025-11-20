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

    // Print a sample logit and the min/max across the vocabulary
    // printf("logit[%d] = %f\n", 0, logits[0]);

    // compute and print min/max logits
    int min_idx = 0, max_idx = 0;
    float min_val = logits[0], max_val = logits[0];
    for (int i = 1; i < VOCAB_SIZE; ++i) {
        if (logits[i] < min_val) {
            min_val = logits[i];
            min_idx = i;
        }
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }

    printf("logits min: idx=%d value=%f\n", min_idx, min_val);
    printf("logits max: idx=%d value=%f\n", max_idx, max_val);

    free(logits);
    free(cache);
    free(m);
    return 0;
}
