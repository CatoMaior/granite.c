/**
 * @file weights_loader.h
 * @brief Header file for loading model weights from files.
 *
 * This file contains the declaration of functions used to initialize
 * a model by loading its weights from specified files.
 */

#pragma once

#include "dtype.h"
#include "model.h"

typedef enum {
    WEIGHTS_TYPE_BF16,
    WEIGHTS_TYPE_Q8_0,
} WeightsType;

/**
 * @brief Load a BF16 tensor from disk into memory.
 *
 * Reads tensor data according to the selected `weights_type`:
 * - WEIGHTS_TYPE_BF16: reads "<tensor_name>_BF16.bin"
 * - WEIGHTS_TYPE_Q8_0: reads "<tensor_name>_Q8_0.bin" and dequantizes to BF16
 *
 * In both cases the output written to `dst` is BF16 with `num_elements`
 * elements. If reading or dequantization fails, the function prints an error
 * and exits the program.
 *
 * @param base_dir Base directory containing the `weights/` subdirectory.
 * @param tensor_name Name of the tensor (used to construct the file path).
 * @param dst Pointer to the destination buffer (must have space for `num_elements` bf16_t values).
 * @param num_elements Number of BF16 elements to read.
 */
void load_tensor_as_bf16(
    const char *base_dir,
    WeightsType weights_type,
    const char *tensor_name,
    bf16_t *dst,
    size_t num_elements);

/**
 * @brief Load an F32 tensor from disk into memory.
 *
 * Opens and reads a binary file containing F32 (float32) tensor data from
 * the path "<base_dir>/weights/<tensor_name>_F32.bin". Reads exactly
 * `num_elements` float values into the destination buffer. If the file cannot
 * be opened or an incomplete read occurs, the function prints an error and
 * exits the program.
 *
 * @param base_dir Base directory containing the `weights/` subdirectory.
 * @param tensor_name Name of the tensor (used to construct the file path).
 * @param dst Pointer to the destination buffer (must have space for `num_elements` floats).
 * @param num_elements Number of float elements to read.
 */
void load_f32_tensor(
    const char *base_dir,
    const char *tensor_name,
    float *dst,
    size_t num_elements);

/**
 * @brief Read and print the first 10 bytes of the file "blk.0.attn_k.weight".
 *
 * Builds the path using the same convention as the loader and `weights_type`.
 * If the file can't be opened or fewer than 10 bytes are available, the
 * function prints a helpful message and returns.
 *
 * @param base_dir Base directory that contains the `weights/` subdirectory.
 */
void print_test_tensor(const char *base_dir, WeightsType weights_type);
