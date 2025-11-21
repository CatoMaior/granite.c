/**
 * @file tokenizer.h
 * @brief Simple tokenizer utilities and detokenizer for the project.
 *
 * This header declares a minimal Tokenizer that holds a mapping from token
 * id to UTF-8 strings, and functions to load a vocabulary file, free the
 * tokenizer, and detokenize an array of token ids into a UTF-8 string.
 */

#pragma once
#include <stddef.h>

/**
 * @brief Minimal tokenizer mapping token ids to UTF-8 strings.
 *
 * - `id_to_token` is an array of `vocab_size` NUL-terminated UTF-8 strings.
 * - `vocab_size` is the number of tokens in the vocabulary.
 */
typedef struct {
    char **id_to_token; /**< array of NUL-terminated UTF-8 token strings */
    int vocab_size;     /**< number of tokens in `id_to_token` */
} Tokenizer;

/**
 * @brief Load a vocabulary file into a tokenizer.
 *
 * The vocabulary file format is project-specific (typically one token per
 * line). The function allocates memory for `tok->id_to_token` and copies
 * strings; the caller is responsible for calling `tokenizer_free` later.
 *
 * @param tok Pointer to an uninitialized Tokenizer structure to fill.
 * @param vocab_path Path to the vocabulary file.
 * @return 0 on success, non-zero on failure.
 */
int tokenizer_load_vocab(Tokenizer *tok, const char *vocab_path);

/**
 * @brief Free memory held by the tokenizer.
 *
 * Frees strings in `id_to_token` and the array itself, and resets fields to
 * a safe empty state.
 *
 * @param tok Pointer to the Tokenizer to free.
 */
void tokenizer_free(Tokenizer *tok);

/**
 * @brief Convert token ids into a UTF-8 string.
 *
 * Concatenates tokens corresponding to `ids[0..n_ids-1]` into the `out`
 * buffer. The function will not write more than `out_size` bytes including
 * the terminating NUL. If the output would be truncated, the resulting
 * string is NUL-terminated but incomplete.
 *
 * @param tok Pointer to an initialized Tokenizer.
 * @param ids Array of token ids to convert.
 * @param n_ids Number of token ids in `ids`.
 * @param out Output buffer to receive the concatenated UTF-8 string.
 * @param out_size Size of the output buffer in bytes.
 */
void detokenize(
    const Tokenizer *tok,
    const int *ids,
    size_t n_ids,
    char *out,
    size_t out_size);
