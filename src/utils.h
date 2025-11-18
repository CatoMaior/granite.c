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
 * This function verifies that the given pointer is not NULL. If the pointer
 * is NULL, it prints an error message to the standard error stream and
 * terminates the program with an exit status of `EXIT_FAILURE`.
 *
 * @param ptr The pointer to check.
 * @param msg The error message to display if the pointer is NULL.
 */
static inline void check_ptr(const void *ptr, const char *msg) {
    if (!ptr) {
        fprintf(stderr, "Error: %s\n", msg);
        exit(EXIT_FAILURE);
    }
}