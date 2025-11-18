/**
 * @file activations.h
 * @brief Activation functions for neural network layers
 *
 * This file contains activation functions used in the transformer model,
 * including SiLU (Sigmoid Linear Unit) and SwiGLU (Swish-Gated Linear Unit).
 */

#pragma once
#include <math.h>
#include <stddef.h>

/**
 * @brief Sigmoid Linear Unit (SiLU) activation function
 *
 * Computes SiLU(x) = x * sigmoid(x) = x / (1 + e^(-x))
 * Also known as Swish activation.
 *
 * @param x Input value
 * @return SiLU activation of x
 */
static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

/**
 * @brief SwiGLU (Swish-Gated Linear Unit) activation
 *
 * Computes element-wise: hidden[i] = SiLU(gate[i]) * up[i]
 * This is a gated activation function commonly used in transformer feed-forward networks.
 *
 * @param hidden Output array of size n (modified in-place)
 * @param gate Gate values array of size n
 * @param up Up-projection values array of size n
 * @param n Number of elements to process
 */
static inline void swiglu(
    float *hidden,
    const float *gate,
    const float *up,
    size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float g = silu(gate[i]);
        hidden[i] = g * up[i];
    }
}
