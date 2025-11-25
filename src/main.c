#define _POSIX_C_SOURCE 199309L
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
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#define NUM_TOKENS 1000
#define BASE_DIR "granite-4.0-350m-BF16"

static int last_stream_lines = 0;
static const char *STREAM_PREFIX = ">> OUTPUT: ";

int get_terminal_width() {
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0) {
        return w.ws_col;
    }
    return 80;
}

int calculate_visual_height(const char *prefix, const char *text, int term_width) {
    int lines = 1;
    int col = 0;

    // Count prefix length
    int prefix_len = strlen(prefix);
    col += prefix_len;

    // If prefix is already longer than width (rare case), handle wrap
    while (col >= term_width) {
        lines++;
        col -= term_width;
    }

    // Now we scan the text character by character
    for (int i = 0; text[i] != '\0'; i++) {
        if (text[i] == '\n') {
            // Explicit line break: new line, column resets to 0
            lines++;
            col = 0;
        } else {
            // Normal character
            col++;
            // If we exceed the width, the terminal wraps automatically
            if (col >= term_width) {
                lines++;
                col = 0; // Or 1, depending on the terminal, but 0 is safe for counting
            }
        }
    }
    return lines;
}

void clear_previous_stream() {
    if (last_stream_lines > 0) {
        for (int i = 0; i < last_stream_lines; i++) {
            // Go up one line and clear it
            // \033[A = Up, \033[2K = Clear Line
            if (i > 0) printf("\033[A");
            printf("\r\033[2K");
        }
        // One last \r for safety
        printf("\r");
    } else {
        // Base case: just clear the current line
        printf("\r\033[2K");
    }
    fflush(stdout);
}

int main(void) {
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

    // tokens[length++] = 100257; // Token ID for BOS (should be not required)
    // tokens[length++] = 13347;  // Token ID for "Hi"
    // tokens[length++] = 0;      // Token ID for "!"
    // tokens[length++] = 2028;   // Token ID for "This"
    // tokens[length++] = 40;     // Token ID for "I"
    // tokens[length++] = 1097;   // Token ID for "Ġam"
    // tokens[length++] = 264;   // Token ID for "Ġa"
    // tokens[length++] = 11190;   // Token ID for "Ġhelpful"
    // tokens[length++] = 220;    // Token ID for "Ġ"
    tokens[length++] = 9906;   // Token ID for "Hello"

    int input_length = length;

    char stream_buf[4096];
    detokenize(&tok, tokens, length, stream_buf, sizeof(stream_buf));

    // Start timing for token generation
    struct timespec start_time, current_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    fprintf(stderr, "Token number | Token id | Hex token id | Throughput (tok/s)\n");
    fprintf(stderr, "-----------------------------------------------------------\n");

    // 5) Generation loop
    for (int pos = 0; pos < NUM_TOKENS + input_length; ++pos) {
        int token_id = tokens[pos];

        clear_previous_stream();

        // Calculate tokens per second
        clock_gettime(CLOCK_MONOTONIC, &current_time);
        float elapsed = (current_time.tv_sec - start_time.tv_sec) +
                        (current_time.tv_nsec - start_time.tv_nsec) / 1e9;

        // Number of tokens generated (excluding input tokens)
        int generated_tokens = (pos >= input_length) ? (pos - input_length + 1) : 0;
        float tokens_per_sec = (elapsed > 0 && generated_tokens > 0) ? generated_tokens / elapsed : 0.0;

        fprintf(stderr, "%12d | %8d |     0x%05x | %18.2f\n",
                pos, token_id, token_id, tokens_per_sec);

        printf("%s%s", STREAM_PREFIX, stream_buf);
        fflush(stdout);

        last_stream_lines = calculate_visual_height(STREAM_PREFIX, stream_buf, get_terminal_width());

        // forward del modello per questo token e posizione
        forward_token(m, cache, token_id, pos, logits);

        // scegli il prossimo token (greedy argmax)
        int next_id = sample_argmax(logits, VOCAB_SIZE);

        // Check for end of text token (100257) and exit loop
        if (next_id == 100257) {
            break;
        }

        tokens[length++] = next_id;

        detokenize(&tok, tokens, length, stream_buf, sizeof(stream_buf));

    }

    printf("\n");

    // 6) Cleanup
    free(logits);
    free(cache);
    tokenizer_free(&tok);
    free(m);

    return 0;
}