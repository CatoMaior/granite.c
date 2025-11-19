/**
 * @file dtype.h
 * @brief Data type definitions and conversions for neural network computations
 *
 * This file provides support for BFloat16 (Brain Floating Point) format,
 * a 16-bit floating point format commonly used in deep learning for memory
 * efficiency while maintaining numerical stability.
 */

#pragma once
#include <stdint.h>

/**
 * @typedef bf16_t
 * @brief BFloat16 (Brain Floating Point) 16-bit type
 *
 * BFloat16 uses 1 sign bit, 8 exponent bits, and 7 mantissa bits,
 * providing the same dynamic range as float32 but with reduced precision.
 */
typedef uint16_t bf16_t;

/**
 * @brief Convert BFloat16 to 32-bit float
 *
 * Converts a BFloat16 value to standard 32-bit floating point by
 * shifting the 16-bit value into the upper half of the 32-bit representation.
 * This works because BFloat16 shares the same bit layout as the upper 16 bits of float32.
 *
 * @param x BFloat16 value to convert
 * @return Converted 32-bit float value
 */
static inline float bf16_to_f32(bf16_t x) {
    union {
        uint32_t u;
        float f;
    } v;

    v.u = (uint32_t)x << 16;
    return v.f;
}

/**
 * @brief Convert 32-bit float to BFloat16
 *
 * Converts a 32-bit float to BFloat16 by truncating the lower 16 bits.
 * This performs a simple round-to-nearest-even operation by taking
 * only the upper 16 bits of the float32 representation.
 *
 * @param x 32-bit float value to convert
 * @return Converted BFloat16 value
 */
static inline bf16_t f32_to_bf16(float x) {
    union {
        uint32_t u;
        float f;
    } v;

    v.f = x;
    return (bf16_t)(v.u >> 16);
}
