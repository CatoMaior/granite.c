#include "weights_loader.h"
#include "utils.h"
#include <linux/limits.h> // PATH_MAX
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

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

void load_bf16_tensor(
    const char *base_dir,
    const char *tensor_name,
    bf16_t *dst,
    size_t num_elements) {
    char path[PATH_MAX];
    make_path(path, sizeof(path), base_dir, tensor_name, "BF16");

    FILE *f = fopen(path, "rb");
    CHECK_PTR(f, "fopen BF16 tensor");

    size_t n_read = fread(dst, sizeof(bf16_t), num_elements, f);
    if (n_read != num_elements) {
        fprintf(stderr, "Incomplete BF16 read (%s): expected %zu, got %zu\n",
                path, num_elements, n_read);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    fclose(f);
}

void print_test_tensor(const char *base_dir) {
    char path[PATH_MAX];
    make_path(path, sizeof(path), base_dir, "blk.0.attn_k.weight", "BF16");

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
