/**
 * @file model_init.h
 * @brief Header file for the model initialization function.
 *
 * This file declares the function used to initialize a Model structure.
 */
#pragma once
#include "model.h"

/**
 * @brief Initializes the given Model structure.
 *
 * This function sets up the necessary components of the provided Model
 * structure to ensure it is ready for use. The specific initialization
 * steps depend on the implementation of the Model.
 *
 * @param m Pointer to the Model structure to be initialized.
 * @return An integer indicating the success or failure of the initialization.
 *         Typically, 0 indicates success, while non-zero values indicate an error.
 */
void model_init(Model *m);

/**
 * @brief Load model weights from a file (placeholder for future implementation)
 *
 * This function will load pre-trained weights from a file into the model.
 * Currently a imports random weights for testing purposes.
 *
 * @param m Pointer to the Model structure to load weights into
 */
void model_load_weights(Model *m);
