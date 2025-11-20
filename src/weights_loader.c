// weights_loader.c
#include "weights_loader.h"
#include "utils.h"
#include <linux/limits.h> // PATH_MAX
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

// base_dir è tipo "granite-4.0-350m-BF16"
static void make_path(
    char *buf, size_t buf_size,
    const char *base_dir,
    const char *tensor_name,
    const char *suffix) // "BF16" o "F32"
{
    // costruiamo: "<base_dir>/weights/<tensor_name>_<suffix>.bin"
    int n = snprintf(
        buf, buf_size,
        "%s/weights/%s_%s.bin",
        base_dir, tensor_name, suffix);
    if (n < 0 || (size_t)n >= buf_size) {
        fprintf(stderr, "Path troppo lungo per tensor '%s'\n", tensor_name);
        exit(EXIT_FAILURE);
    }
}

static void load_bf16_tensor(
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

static void load_f32_tensor(
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

void model_init_from_dir(Model *m, const char *base_dir) {
    // azzera tutto
    memset(m, 0, sizeof(Model));

    // set meta (non vengono dal file ma dai metadati che già conosci)
    m->vocab_size = VOCAB_SIZE;
    m->context_length = CONTEXT_LENGTH;
    m->n_layers = N_LAYERS;
    m->d_model = D_MODEL;
    m->n_heads = N_HEADS;
    m->n_kv_heads = N_KV_HEADS;
    m->head_dim = HEAD_DIM;
    m->d_ff = D_FF;

    // 1) token embedding BF16: [D_MODEL, VOCAB_SIZE]
    load_bf16_tensor(
        base_dir,
        "token_embd.weight",
        &m->token_embd[0][0],
        (size_t)D_MODEL * (size_t)VOCAB_SIZE);

    // 2) output_norm F32: [D_MODEL]
    load_f32_tensor(
        base_dir,
        "output_norm.weight",
        m->output_norm,
        (size_t)D_MODEL);

    // 3) layers blk.0 ... blk.27
    for (int l = 0; l < N_LAYERS; ++l) {
        Layer *L = &m->layers[l];

        char name[64];

        // --- Norm F32 ---
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", l);
        load_f32_tensor(
            base_dir,
            name,
            L->attn_norm,
            (size_t)D_MODEL);

        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", l);
        load_f32_tensor(
            base_dir,
            name,
            L->ffn_norm,
            (size_t)D_MODEL);

        // --- Attention BF16 ---

        // attn_q.weight [1024,1024]
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
        load_bf16_tensor(
            base_dir,
            name,
            &L->w_q[0][0],
            (size_t)D_MODEL * (size_t)D_MODEL);

        // attn_k.weight [1024,256]
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
        load_bf16_tensor(
            base_dir,
            name,
            &L->w_k[0][0],
            (size_t)D_MODEL * (size_t)(N_KV_HEADS * HEAD_DIM));

        // attn_v.weight [1024,256]
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
        load_bf16_tensor(
            base_dir,
            name,
            &L->w_v[0][0],
            (size_t)D_MODEL * (size_t)(N_KV_HEADS * HEAD_DIM));

        // attn_output.weight [1024,1024]
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
        load_bf16_tensor(
            base_dir,
            name,
            &L->w_o[0][0],
            (size_t)D_MODEL * (size_t)D_MODEL);

        // --- MLP BF16 ---

        // ffn_gate.weight [1024,2048]
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", l);
        load_bf16_tensor(
            base_dir,
            name,
            &L->w_gate[0][0],
            (size_t)D_MODEL * (size_t)D_FF);

        // ffn_up.weight [1024,2048]
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", l);
        load_bf16_tensor(
            base_dir,
            name,
            &L->w_up[0][0],
            (size_t)D_MODEL * (size_t)D_FF);

        // ffn_down.weight [2048,1024]
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", l);
        load_bf16_tensor(
            base_dir,
            name,
            &L->w_down[0][0],
            (size_t)D_FF * (size_t)D_MODEL);
    }
}
