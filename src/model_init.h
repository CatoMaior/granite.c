/**
 * @file model_init_rnd.h
 * @brief Header file for the model initialization function.
 *
 * This file declares the function used to initialize a Model structure.
 */
#pragma once
#include "model.h"

/**
 * @brief Initialize a Model by loading weights from a directory on disk.
 *
 * This function zeroes the model structure, sets its metadata (dimensions,
 * vocabulary size, etc.), and then loads all required weight tensors from
 * the specified base directory. The directory is expected to contain a
 * `weights/` subdirectory and binary files with names following the pattern:
 *
 *   <tensor_name>_<SUFFIX>.bin
 *
 * where <SUFFIX> is either BF16 or F32 depending on the tensor precision.
 * For example: `token_embd.weight_BF16.bin`, `output_norm.weight_F32.bin`,
 * `blk.0.attn_q.weight_BF16.bin`, and so on.
 *
 * On any I/O error or incomplete read, the function prints an error message
 * to stderr and terminates the program.
 *
 * @param m Pointer to the Model to initialize.
 * @param base_dir Base directory containing the `weights/` folder and weight files.
 */
void model_init_from_dir(Model *m, const char *base_dir);

/**
 * @brief Set the static metadata fields for a Model.
 *
 * This helper fills the model's metadata fields (such as vocabulary size,
 * context length, number of layers, model dimensions, and related constants)
 * based on the project's compile-time constants. It does not allocate or
 * initialize weight arrays; it only sets the metadata that other initialization
 * code expects to be present before loading weights.
 *
 * @param m Pointer to the Model structure to populate with metadata.
 */
void model_set_meta(Model *m);

/**
 * @brief Initializes the given Model structure with random weights.
 * For testing purposes only.
 *
 * @param m Pointer to the Model structure to be initialized.
 * @return An integer indicating the success or failure of the initialization.
 *         Typically, 0 indicates success, while non-zero values indicate an error.
 */
void model_init_rnd(Model *m);

/**
 * @brief Randomly initialize all weights in the Model structure.
 *
 * This function will load random weights into the model for testing purposes.
 *
 * @param m Pointer to the Model structure to load weights into
 */
void model_set_rnd_weights(Model *m);
