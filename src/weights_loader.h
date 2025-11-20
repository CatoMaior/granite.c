/**
 * @file weights_loader.h
 * @brief Header file for loading model weights from a directory.
 *
 * This file contains the declaration of functions used to initialize
 * a model by loading its weights from a specified directory.
 */

#pragma once

#include "dtype.h"
#include "model.h"

/**
 * @brief Initializes a model by loading weights from a directory.
 *
 * This function loads the weights of the model from the specified base directory
 * and initializes the model structure accordingly.
 *
 * @param m Pointer to the Model structure to be initialized.
 * @param base_dir Path to the base directory containing the model weights.
 */
void model_init_from_dir(Model *m, const char *base_dir);

/**
 * @brief Read and print the first 10 bytes of the file "blk.0.attn_k.weight".
 *
 * Builds the path using the same convention as the loader (suffix "BF16").
 * If the file can't be opened or fewer than 10 bytes are available, the
 * function prints a helpful message and returns.
 *
 * @param base_dir Base directory that contains the `weights/` subdirectory.
 */
void print_test_tensor(const char *base_dir);
