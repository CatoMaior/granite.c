#include "weights_loader.h"
#include "utils.h"
#include <linux/limits.h> // PATH_MAX
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define Q8_0_BLOCK_SIZE 32
#define Q8_0_BLOCK_BYTES 34

static float f16_to_f32(uint16_t h) {
    // IEEE-754 half precision to float32 conversion.
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    uint32_t out;

    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            // Subnormal: normalize mantissa.
            int e = -14;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x03ff;
            out = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        // Inf/NaN
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        // Normalized number.
        int e = (int)exp - 15 + 127;
        out = sign | ((uint32_t)e << 23) | (mant << 13);
    }

    union {
        uint32_t u;
        float f;
    } v;
    v.u = out;
    return v.f;
}

static void make_path(
    char *buf, size_t buf_size,
    const char *base_dir, // base_dir is like "granite-4.0-350m-BF16"
    const char *tensor_name,
    const char *suffix) // "BF16" o "F32"
{
    // build path: "<base_dir>/weights/<tensor_name>_<suffix>.bin"
    int n = snprintf(
        buf, buf_size,
        "%s/weights/%s_%s.bin",
        base_dir, tensor_name, suffix);
    if (n < 0 || (size_t)n >= buf_size) {
        fprintf(stderr, "Path troppo lungo per tensor '%s'\n", tensor_name);
        exit(EXIT_FAILURE);
    }
}

void load_tensor_as_bf16(
    const char *base_dir,
    WeightsType weights_type,
    const char *tensor_name,
    bf16_t *dst,
    size_t num_elements) {
    if (weights_type == WEIGHTS_TYPE_BF16) {
        char path_bf16[PATH_MAX];
        make_path(path_bf16, sizeof(path_bf16), base_dir, tensor_name, "BF16");

        FILE *f = fopen(path_bf16, "rb");
        CHECK_PTR(f, "fopen BF16 tensor");

        size_t n_read = fread(dst, sizeof(bf16_t), num_elements, f);
        if (n_read != num_elements) {
            fprintf(stderr, "Incomplete BF16 read (%s): expected %zu, got %zu\n",
                    path_bf16, num_elements, n_read);
            fclose(f);
            exit(EXIT_FAILURE);
        }

        fclose(f);
        return;
    }

    if (weights_type != WEIGHTS_TYPE_Q8_0) {
        fprintf(stderr, "Unsupported weights type for BF16 tensor load: %d\n", (int)weights_type);
        exit(EXIT_FAILURE);
    }

    // Q8_0 path: load blocks and dequantize directly into BF16.
    char path_q8_0[PATH_MAX];
    make_path(path_q8_0, sizeof(path_q8_0), base_dir, tensor_name, "Q8_0");

    FILE *f = fopen(path_q8_0, "rb");
    CHECK_PTR(f, "fopen Q8_0 tensor");

    if (num_elements % Q8_0_BLOCK_SIZE != 0) {
        fprintf(stderr,
                "Q8_0 tensor element count must be multiple of %d (%s): got %zu\n",
                Q8_0_BLOCK_SIZE,
                path_q8_0,
                num_elements);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    const size_t n_blocks = num_elements / Q8_0_BLOCK_SIZE;
    uint8_t block[Q8_0_BLOCK_BYTES];
    size_t out_idx = 0;

    for (size_t b = 0; b < n_blocks; ++b) {
        size_t got = fread(block, 1, Q8_0_BLOCK_BYTES, f);
        if (got != Q8_0_BLOCK_BYTES) {
            fprintf(stderr,
                    "Incomplete Q8_0 read (%s): block %zu/%zu\n",
                    path_q8_0,
                    b,
                    n_blocks);
            fclose(f);
            exit(EXIT_FAILURE);
        }

        uint16_t d_bits = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
        float d = f16_to_f32(d_bits);

        for (size_t i = 0; i < Q8_0_BLOCK_SIZE; ++i) {
            int8_t q = (int8_t)block[2 + i];
            float v = d * (float)q;
            dst[out_idx++] = f32_to_bf16(v);
        }
    }

    // Ensure there is no trailing data in the file.
    int extra = fgetc(f);
    if (extra != EOF) {
        fprintf(stderr,
                "Q8_0 tensor has trailing data (%s)\n",
                path_q8_0);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    fclose(f);
}

void print_test_tensor(const char *base_dir, WeightsType weights_type) {
    char path[PATH_MAX];
    if (weights_type == WEIGHTS_TYPE_BF16) {
        make_path(path, sizeof(path), base_dir, "blk.0.attn_k.weight", "BF16");
    } else if (weights_type == WEIGHTS_TYPE_Q8_0) {
        make_path(path, sizeof(path), base_dir, "blk.0.attn_k.weight", "Q8_0");
    } else {
        fprintf(stderr, "Unsupported weights type for test tensor print: %d\n", (int)weights_type);
        exit(EXIT_FAILURE);
    }

    FILE *f = fopen(path, "rb");
    CHECK_PTR(f, "fopen attn_k.weight for byte print");

    unsigned char buf[10];
    size_t n = fread(buf, 1, sizeof(buf), f);
    if (n == 0) {
        fprintf(stderr, "No bytes read from %s\n", path);
        fclose(f);
        return;
    }

    printf("First %zu bytes of %s:\n", n, path);
    for (size_t i = 0; i < n; ++i) {
        printf("0x%02x ", buf[i]);
    }
    printf("\n");

    fclose(f);
}

void load_f32_tensor(
    const char *base_dir,
    const char *tensor_name,
    float *dst,
    size_t num_elements) {
    char path[PATH_MAX];
    make_path(path, sizeof(path), base_dir, tensor_name, "F32");

    FILE *f = fopen(path, "rb");
    CHECK_PTR(f, "fopen F32 tensor");

    size_t n_read = fread(dst, sizeof(float), num_elements, f);
    if (n_read != num_elements) {
        fprintf(stderr, "Incomplete F32 read (%s): expected %zu, got %zu\n",
                path, num_elements, n_read);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    fclose(f);
}
