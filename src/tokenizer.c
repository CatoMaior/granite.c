#include "tokenizer.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *SPACE_PREFIX = "Ġ";  // U+0120
static const char *NEWLINE_TOKEN = "Ċ"; // U+010A

// helper to read a line from file and strip newline
static char *read_line_strip(FILE *f) {
    size_t cap = 256;
    size_t len = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;

    int c;
    while ((c = fgetc(f)) != EOF) {
        if (c == '\n') break;
        if (len + 1 >= cap) {
            cap *= 2;
            char *nb = realloc(buf, cap);
            if (!nb) {
                free(buf);
                return NULL;
            }
            buf = nb;
        }
        buf[len++] = (char)c;
    }
    if (len == 0 && c == EOF) {
        free(buf);
        return NULL; // end of file
    }
    buf[len] = '\0';
    return buf;
}

int tokenizer_load_vocab(Tokenizer *tok, const char *vocab_path) {
    FILE *f = fopen(vocab_path, "r");
    CHECK_PTR(f, "fopen vocab file");

    // count lines to get vocab size
    int lines = 0;
    {
        int c;
        int in_line = 0;
        while ((c = fgetc(f)) != EOF) {
            if (c == '\n') {
                lines++;
                in_line = 0;
            } else if (!in_line) {
                in_line = 1;
            }
        }
        // if file does not end with newline, count last line
        if (in_line) lines++;
        rewind(f);
    }

    tok->vocab_size = lines;
    tok->id_to_token = calloc(lines, sizeof(char *));
    CHECK_PTR(tok->id_to_token, "calloc id_to_token");

    for (int i = 0; i < lines; ++i) {
        char *line = read_line_strip(f);
        CHECK_PTR(line, "read_line_strip vocab");
        tok->id_to_token[i] = line;
    }

    fclose(f);
    return 0;
}

void tokenizer_free(Tokenizer *tok) {
    if (!tok || !tok->id_to_token) return;
    for (int i = 0; i < tok->vocab_size; ++i) {
        free(tok->id_to_token[i]);
    }
    free(tok->id_to_token);
    tok->id_to_token = NULL;
    tok->vocab_size = 0;
}

void detokenize(
    const Tokenizer *tok,
    const int *ids,
    size_t n_ids,
    char *out,
    size_t out_size) {
    size_t pos = 0;
    if (out_size == 0) return;

    for (size_t t = 0; t < n_ids; ++t) {
        int id = ids[t];
        if (id < 0 || id >= tok->vocab_size) {
            // unknown token: skip or put something
            const char *unk = "<UNK>";
            size_t len = strlen(unk);
            if (pos + len + 1 >= out_size) break;
            memcpy(out + pos, unk, len);
            pos += len;
            continue;
        }

        const char *piece = tok->id_to_token[id];

        // handle newline tokens within the piece (up to 12)
        const char *newline_pos;
        const char *search_start = piece;
        int newline_count = 0;

        while ((newline_pos = strstr(search_start, NEWLINE_TOKEN)) != NULL && newline_count < 12) {
            // Write everything before this newline token
            size_t prefix_len = newline_pos - search_start;
            if (prefix_len > 0) {
                // Handle space tokens within this segment (up to 128)
                const char *space_pos;
                const char *space_search = search_start;
                int space_count = 0;

                while ((space_pos = strstr(space_search, SPACE_PREFIX)) != NULL &&
                       space_pos < newline_pos && space_count < 128) {
                    // Write everything before this space token
                    size_t before_space_len = space_pos - space_search;
                    if (before_space_len > 0) {
                        if (pos + before_space_len >= out_size) break;
                        memcpy(out + pos, space_search, before_space_len);
                        pos += before_space_len;
                    }

                    // Write the space character
                    if (pos + 1 >= out_size) break;
                    out[pos++] = ' ';

                    // Move past this space token
                    space_search = space_pos + strlen(SPACE_PREFIX);
                    space_count++;
                }

                // Write remaining text after last space token (or all if no space tokens)
                size_t remaining_before_newline = newline_pos - space_search;
                if (remaining_before_newline > 0) {
                    if (pos + remaining_before_newline >= out_size) break;
                    memcpy(out + pos, space_search, remaining_before_newline);
                    pos += remaining_before_newline;
                }
            }

            // Write the newline character
            if (pos + 1 >= out_size) break;
            out[pos++] = '\n';

            // Move past this newline token
            search_start = newline_pos + strlen(NEWLINE_TOKEN);
            newline_count++;
        }

        // If we found newline tokens, handle the remainder after the last one
        if (newline_count > 0) {
            // Handle space tokens in the remainder (up to 128)
            const char *space_pos;
            const char *space_search = search_start;
            int space_count = 0;

            while ((space_pos = strstr(space_search, SPACE_PREFIX)) != NULL && space_count < 128) {
                // Write everything before this space token
                size_t before_space_len = space_pos - space_search;
                if (before_space_len > 0) {
                    if (pos + before_space_len >= out_size) break;
                    memcpy(out + pos, space_search, before_space_len);
                    pos += before_space_len;
                }

                // Write the space character
                if (pos + 1 >= out_size) break;
                out[pos++] = ' ';

                // Move past this space token
                space_search = space_pos + strlen(SPACE_PREFIX);
                space_count++;
            }

            // Write remaining text after last space token
            size_t remaining_len = strlen(space_search);
            if (remaining_len > 0) {
                if (pos + remaining_len >= out_size) break;
                memcpy(out + pos, space_search, remaining_len);
                pos += remaining_len;
            }
            continue;
        }

        // No newline tokens found - handle space tokens (up to 128)
        const char *space_pos;
        const char *space_search = piece;
        int space_count = 0;

        while ((space_pos = strstr(space_search, SPACE_PREFIX)) != NULL && space_count < 128) {
            // Write everything before this space token
            size_t before_space_len = space_pos - space_search;
            if (before_space_len > 0) {
                if (pos + before_space_len >= out_size) break;
                memcpy(out + pos, space_search, before_space_len);
                pos += before_space_len;
            }

            // Write the space character
            if (pos + 1 >= out_size) break;
            out[pos++] = ' ';

            // Move past this space token
            space_search = space_pos + strlen(SPACE_PREFIX);
            space_count++;
        }

        // Write remaining text after last space token (or all if no space tokens found)
        size_t remaining_len = strlen(space_search);
        if (remaining_len > 0) {
            if (pos + remaining_len >= out_size) break;
            memcpy(out + pos, space_search, remaining_len);
            pos += remaining_len;
        }
    }

    if (pos >= out_size) pos = out_size - 1;
    out[pos] = '\0';
}