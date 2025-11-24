#include "forward.h"
#include "math_utils.h"
#include "model.h"
#include "model_init.h"
#include "tokenizer.h"
#include "utils.h"
#include "weights_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_TOKENS 15
#define BASE_DIR "granite-4.0-350m-BF16"

int main(void) {
    // On heap because model is too large for stack
    Model *m = malloc(sizeof(Model));
    CHECK_PTR(m, "malloc model");

    print_test_tensor(BASE_DIR);

    model_init_from_dir(m, BASE_DIR);

    Tokenizer tok = {0};
    char vocab_path[256];
    snprintf(vocab_path, sizeof(vocab_path), "%s/vocab.txt", BASE_DIR);
    if (tokenizer_load_vocab(&tok, vocab_path) != 0) {
        fprintf(stderr, "Error loading vocab\n");
        return 1;
    }

    KVCache *cache = malloc(sizeof(KVCache));
    CHECK_PTR(cache, "malloc KVCache");
    memset(cache, 0, sizeof(KVCache));

    float *logits = malloc(sizeof(float) * VOCAB_SIZE);
    CHECK_PTR(logits, "malloc logits");

    int tokens[NUM_TOKENS + 3];
    int length = 0;

    // tokens[length++] = 100257; // Token ID for BOS (should be not required)
    // tokens[length++] = 13347;  // Token ID for "Hi"
    // tokens[length++] = 0;      // Token ID for "!"
    // tokens[length++] = 2028;   // Token ID for "This"
    // tokens[length++] = 40;     // Token ID for "I"
    // tokens[length++] = 1097;   // Token ID for "Ġam"
    // tokens[length++] = 220;    // Token ID for "Ġ"
    tokens[length++] = 9906;   // Token ID for "Hello"

    int input_length = length;

    // 5) Generation loop
    for (int pos = 0; pos < NUM_TOKENS + input_length; ++pos) {
        int token_id = tokens[pos];

        printf("[pos=%d] input token_id = %d\n", pos, token_id);

        // forward del modello per questo token e posizione
        forward_token(m, cache, token_id, pos, logits);

        // scegli il prossimo token (greedy argmax)
        int next_id = sample_argmax(logits, VOCAB_SIZE);

        tokens[length++] = next_id;
    }

    // 6) Detokenize the sequence
    char decoded[4096];
    int start_idx = 0;
    size_t n_ids = (size_t)(length - start_idx);

    detokenize(&tok, &tokens[start_idx], n_ids, decoded, sizeof(decoded));

    printf("\n=== Token sequence ===\n");
    for (int i = 0; i < length; ++i) {
        printf("%d ", tokens[i]);
    }
    printf("\n\n=== Decoded Text ===\n");
    printf("%s\n", decoded);

    // 7) Cleanup
    free(logits);
    free(cache);
    tokenizer_free(&tok);
    free(m);

    return 0;
}
