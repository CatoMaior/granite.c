#include "console_utils.h"
#include "forward.h"
#include "math_utils.h"
#include "model.h"
#include "model_init.h"
#include "timing.h"
#include "tokenizer.h"
#include "utils.h"
#include "weights_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define NUM_TOKENS 1000
#define BASE_DIR "granite-4.0-350m-BF16"

static const char *STREAM_PREFIX = ">> OUTPUT: ";

int main(void) {
    // Initialize RoPE cache with precomputed cos/sin values
    rope_cache_init();

    // On heap because model is too large for stack
    Model *m = malloc(sizeof(Model));
    CHECK_PTR(m, "malloc model");

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

    // --- Something to input ---
    // tokens[length++] = 100257; // <|end_of_text|>
    // tokens[length++] = 13347;  // "Hi"
    // tokens[length++] = 0;      // "!"
    // tokens[length++] = 2028;   // "This"
    // tokens[length++] = 40;     // "I"
    // tokens[length++] = 1097;   // "Ġam"
    // tokens[length++] = 264;    // "Ġa"
    // tokens[length++] = 11190;  // "Ġhelpful"
    // tokens[length++] = 220;    // "Ġ"
    tokens[length++] = 9906;   // "Hello"

    // --- Standard chat sysprompt ---
    // tokens[length++] = 100264; // <|start_of_role|>
    // tokens[length++] = 9125;   // "system"
    // tokens[length++] = 100265; // <|end_of_role|>
    // tokens[length++] = 2675;   // "You"
    // tokens[length++] = 527;    // "Ġare"
    // tokens[length++] = 264;    // "Ġa"
    // tokens[length++] = 11190;  // "Ġhelpful"
    // tokens[length++] = 18328;  // "Ġassistant"
    // tokens[length++] = 13;     // "."
    // tokens[length++] = 5321;   // "ĠPlease"
    // tokens[length++] = 6106;   // "Ġensure"
    // tokens[length++] = 14847;  // "Ġresponses"
    // tokens[length++] = 527;    // "Ġare"
    // tokens[length++] = 6721;   // "Ġprofessional"
    // tokens[length++] = 11;     // ","
    // tokens[length++] = 13687;  // "Ġaccurate"
    // tokens[length++] = 11;     // ","
    // tokens[length++] = 323;    // "Ġand"
    // tokens[length++] = 6220;   // "Ġsafe"
    // tokens[length++] = 13;     // "."
    // tokens[length++] = 100257; // <|end_of_text|>
    // tokens[length++] = 198;    // \n
    // tokens[length++] = 100264; // <|start_of_role|>
    // tokens[length++] = 882;    // "user"
    // tokens[length++] = 100265; // <|end_of_role|>
    // tokens[length++] = 9906;   // "Hello"
    // tokens[length++] = 100257; // <|end_of_text|>
    // tokens[length++] = 198;    // \n
    // tokens[length++] = 100264; // <|start_of_role|>
    // tokens[length++] = 78191;  // "assistant"
    // tokens[length++] = 100265; // <|end_of_role|>

    int input_length = length;

    char stream_buf[4096];
    detokenize(&tok, tokens, length, stream_buf, sizeof(stream_buf));

    // Initialize throughput tracking
    ThroughputTracker timing;
    timing_init(&timing);

    // Initialize console streaming state
    StreamState console_state = {0};

    fprintf(stderr, "Token number | Token id | Hex token id | Overall (tok/s) | Last 4 (tok/s)\n");
    fprintf(stderr, "-------------------------------------------------------------------------\n");

    // 5) Generation loop
    for (int pos = 0; pos < NUM_TOKENS + input_length; ++pos) {
        int token_id = tokens[pos];

        console_clear_previous_stream(&console_state);

        // Number of tokens generated (excluding input tokens)
        int generated_tokens = (pos >= input_length) ? (pos - input_length + 1) : 0;

        // Calculate throughput metrics
        float overall_tps, window_tps;
        timing_record_token(&timing, generated_tokens, &overall_tps, &window_tps);

        fprintf(stderr, "%12d | %8d |      0x%05x | %15.2f | %14.2f\n",
                pos, token_id, token_id, overall_tps, window_tps);

        // Check for end of text token (100257) and exit loop
        if (token_id == 100257 && pos >= input_length) break;

        detokenize(&tok, tokens, length, stream_buf, sizeof(stream_buf));
        printf("%s%s", STREAM_PREFIX, stream_buf);
        fflush(stdout);

        console_update_stream_state(&console_state, STREAM_PREFIX, stream_buf,
                                     console_get_terminal_width());

        // Forward del modello per questo token e posizione
        forward_token(m, cache, token_id, pos, logits);

        // Choose next token (greedy argmax)
        int next_id = sample_argmax(logits, VOCAB_SIZE);

        // If input parsing is done, append the generated token
        if (pos >= input_length - 1) tokens[length++] = next_id;
    }

    printf("%s%s\n", STREAM_PREFIX, stream_buf);

    // 6) Cleanup
    free(logits);
    free(cache);
    tokenizer_free(&tok);
    free(m);

    return 0;
}