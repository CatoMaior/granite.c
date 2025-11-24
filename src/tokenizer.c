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

        // handle some special tokens
        if (strcmp(piece, NEWLINE_TOKEN) == 0) {
            if (pos + 1 >= out_size) break;
            out[pos++] = '\n';
            continue;
        }

        // if starts with "Ġ", write space and then the rest of the piece
        if (strncmp(piece, SPACE_PREFIX, strlen(SPACE_PREFIX)) == 0) {
            if (pos + 1 >= out_size) break;
            out[pos++] = ' ';
            piece += strlen(SPACE_PREFIX); // skip prefix
        }

        size_t len = strlen(piece);
        if (pos + len >= out_size) break;
        memcpy(out + pos, piece, len);
        pos += len;
    }

    if (pos >= out_size) pos = out_size - 1;
    out[pos] = '\0';
}