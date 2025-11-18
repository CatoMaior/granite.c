#include <stdlib.h>
#include <stdio.h>
#include "forward.h"
#include "model.h"
#include "utils.h"
#include "model_init.h"

int main(void) {
    // On heap because model is too large for stack
    Model *m = malloc(sizeof(Model));
    check_ptr(m, "Failed to allocate memory for model\n");
    model_init(m);

    float *logits = malloc(sizeof(float) * VOCAB_SIZE);
    check_ptr(logits, "Failed to allocate memory for logits\n");

    forward_token(m, 0, 0, logits);

    for (int i = 0; i < 10; ++i) {
        printf("logit[%d] = %f\n", i, logits[i]);
    }

    free(logits);
    free(m);
    return 0;
}
