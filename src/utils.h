/**
 * @file utils.h
 * @brief Utility functions for common operations.
 *
 * This header file provides utility functions that can be used
 * throughout the program to perform common tasks such as pointer
 * validation.
 */
#pragma once
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Checks if a pointer is valid and exits the program if it is not.
 *
 * This macro verifies that the given pointer is not NULL. If the pointer
 * is NULL, it prints an error message to the standard error stream and
 * terminates the program with an exit status of `EXIT_FAILURE`.
 *
 * @param ptr The pointer to check.
 * @param msg The error message to display if the pointer is NULL.
 */
#define CHECK_PTR(ptr, msg) \
    do { \
        if (!(ptr)) { \
            fprintf(stderr, "Error: %s\n", (msg)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/**
 * @brief Select the index of the largest logit (argmax sampling).
 *
 * This inline helper returns the index of the maximum value in the
 * `logits` array. It implements a deterministic argmax selection used when
 * sampling by highest score rather than probabilistic sampling.
 *
 * Contract:
 * - Inputs: `logits` must point to an array of at least `vocab_size`
 *   floats. `vocab_size` must be >= 1.
 * - Output: returns the index in range [0, vocab_size) corresponding to the
 *   maximum value. If multiple entries are equal to the maximum, the first
 *   (lowest index) is returned.
 *
 * @param logits Pointer to array of float scores/logits.
 * @param vocab_size Number of elements in `logits` (must be >= 1).
 * @return Index of the maximum logit.
 */
static inline int sample_argmax(const float *logits, int vocab_size) {
    int best_id = 0;
    float best_val = logits[0];

    for (int i = 1; i < vocab_size; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_id = i;
        }
    }
    return best_id;
}
