/**
 * @file math_utils.h
 * @brief Mathematical utility functions for neural network operations
 *
 * This file contains inline mathematical utility functions for vector operations,
 * matrix-vector multiplication, and normalization used in the transformer model.
 */

#pragma once
#include <math.h>
#include <stddef.h>

/**
 * @brief Copy a vector from source to destination
 *
 * @param dst Destination vector
 * @param src Source vector
 * @param n Number of elements to copy
 */
static inline void vec_copy(float *dst, const float *src, size_t n) {
    for (size_t i = 0; i < n; ++i)
        dst[i] = src[i];
}

/**
 * @brief Add vector x to vector y in-place (y = y + x)
 *
 * @param y Destination vector (modified in-place)
 * @param x Source vector to add
 * @param n Number of elements
 */
static inline void vec_add_inplace(float *y, const float *x, size_t n) {
    for (size_t i = 0; i < n; ++i)
        y[i] += x[i];
}

/**
 * @brief Scale a vector in-place by a scalar (y = y * s)
 *
 * @param y Vector to scale (modified in-place)
 * @param s Scalar multiplier
 * @param n Number of elements
 */
static inline void vec_scale_inplace(float *y, float s, size_t n) {
    for (size_t i = 0; i < n; ++i)
        y[i] *= s;
}

/**
 * @brief Matrix-vector multiplication
 *
 * Computes out[j] = sum_i x[i] * W[i,j] where W is stored in row-major format.
 *
 * @param out Output vector of size out_dim
 * @param mat Input matrix of size [in_dim x out_dim] in row-major order
 * @param x Input vector of size in_dim
 * @param in_dim Input dimension (number of rows in matrix)
 * @param out_dim Output dimension (number of columns in matrix)
 */
static inline void matvec(
    float *out,
    const float *mat,
    const float *x,
    size_t in_dim,
    size_t out_dim) {
    for (size_t j = 0; j < out_dim; ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < in_dim; ++i) {
            sum += mat[i * out_dim + j] * x[i];
        }
        out[j] = sum;
    }
}

/**
 * @brief Root Mean Square (RMS) normalization
 *
 * Normalizes the input vector using RMS normalization and applies
 * element-wise scaling with weights.
 *
 * Formula: y[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]
 *
 * @param y Output normalized vector of size n
 * @param x Input vector of size n
 * @param w Weight vector (gamma) of size n for element-wise scaling
 * @param n Number of elements
 * @param eps Epsilon value for numerical stability
 */
static inline void rms_norm(
    float *y,
    const float *x,
    const float *w, // gamma
    size_t n,
    float eps) {
    float mean_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        mean_sq += x[i] * x[i];
    }
    mean_sq /= (float)n;
    float inv_rms = 1.0f / sqrtf(mean_sq + eps);
    for (size_t i = 0; i < n; ++i) {
        y[i] = x[i] * inv_rms * w[i];
    }
}

/**
 * @brief Softmax activation function (in-place)
 *
 * Computes the softmax function on the input vector in-place, converting
 * the values to a probability distribution that sums to 1.0.
 * Uses the numerically stable formulation: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
 *
 * Formula: x[i] = exp(x[i] - max(x)) / sum_j(exp(x[j] - max(x)))
 *
 * @param x Input/output vector of size n (modified in-place)
 * @param n Number of elements
 */
static inline void softmax_inplace(float *x, size_t n) {
    if (n == 0) return;

    float maxv = x[0];
    for (size_t i = 1; i < n; ++i) {
        if (x[i] > maxv) maxv = x[i];
    }

    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        x[i] = expf(x[i] - maxv);
        sum += x[i];
    }

    float inv = 1.0f / sum;
    for (size_t i = 0; i < n; ++i) {
        x[i] *= inv;
    }
}
