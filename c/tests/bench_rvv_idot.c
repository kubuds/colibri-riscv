/* Real-kernel throughput for idot.h on SpacemiT K3, same source built twice:
 *   -mcpu=spacemit-x100  -> the __riscv_vector tiers
 *   -march=rv64gc        -> the scalar fallback
 * Exactness of the RVV tiers is gated by tests/test_idot.c; this measures
 * the kernels the engines actually call (dot_i8i8, dot_i8i8_rows, dot_i4i8,
 * qrow_i8) at one activation row per call, single core, warm caches.
 */
#include "../idot.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static uint32_t rs = 0x12345678u;
static uint32_t xr(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }

#define I_DIM 4096
#define N_REPEAT 20000

static int8_t  W[I_DIM + 64] __attribute__((aligned(64)));
static int8_t  X[I_DIM + 64] __attribute__((aligned(64)));
static uint8_t W4[I_DIM / 2 + 64] __attribute__((aligned(64)));
static int8_t  XQ[I_DIM + 64] __attribute__((aligned(64)));
static float   XF[I_DIM + 64] __attribute__((aligned(64)));

static double bench_i8(int32_t (*f)(const int8_t *, const int8_t *, int)) {
    int I = I_DIM;
    for (int i = 0; i < I; i++) X[i] = (int8_t)((int)(xr() % 255) - 127);
    for (int i = 0; i < I; i++) W[i] = (int8_t)((int)(xr() % 256) - 128);
    volatile int32_t sink = 0;
    uint64_t best = ~0ull;
    for (int rep = 0; rep < 3; rep++) {
        uint64_t t0 = ns_now();
        for (int r = 0; r < N_REPEAT; r++) sink += f(W, X, I);
        uint64_t t1 = ns_now();
        if (t1 - t0 < best) best = t1 - t0;
    }
    (void)sink;
    return (double)best / N_REPEAT;
}

static double bench_rows(void (*f)(const int8_t *, const int8_t *, int, int, int32_t *), int n) {
    int I = I_DIM;
    int32_t out[4];
    volatile int32_t sink = 0;
    uint64_t best = ~0ull;
    for (int rep = 0; rep < 3; rep++) {
        uint64_t t0 = ns_now();
        for (int r = 0; r < N_REPEAT; r++) { f(W, X, I, n, out); sink += out[0]; }
        uint64_t t1 = ns_now();
        if (t1 - t0 < best) best = t1 - t0;
    }
    (void)sink;
    return (double)best / N_REPEAT;
}

static double bench_i4(int32_t (*f)(const uint8_t *, const int8_t *, int)) {
    int I = I_DIM;
    for (int i = 0; i < I; i++) X[i] = (int8_t)((int)(xr() % 255) - 127);
    for (int i = 0; i < (I + 1) / 2; i++) W4[i] = (uint8_t)(xr() & 0xFF);
    volatile int32_t sink = 0;
    uint64_t best = ~0ull;
    for (int rep = 0; rep < 3; rep++) {
        uint64_t t0 = ns_now();
        for (int r = 0; r < N_REPEAT; r++) sink += f(W4, X, I);
        uint64_t t1 = ns_now();
        if (t1 - t0 < best) best = t1 - t0;
    }
    (void)sink;
    return (double)best / N_REPEAT;
}

static double bench_qrow(float (*f)(const float *, int8_t *, int)) {
    int I = I_DIM;
    volatile float sink = 0;
    uint64_t best = ~0ull;
    for (int rep = 0; rep < 3; rep++) {
        uint64_t t0 = ns_now();
        for (int r = 0; r < N_REPEAT; r++) sink += f(XF, XQ, I);
        uint64_t t1 = ns_now();
        if (t1 - t0 < best) best = t1 - t0;
    }
    (void)sink;
    return (double)best / N_REPEAT;
}

int main(void) {
    const char *tier = IDOT_KERNEL;
    for (int i = 0; i < I_DIM; i++) XF[i] = ((float)(xr() % 2001) - 1000.f) / 500.f;
    for (int i = 0; i < I_DIM; i++) { X[i] = (int8_t)((int)(xr() % 255) - 127); W[i] = (int8_t)((int)(xr() % 256) - 128); }
    for (int i = 0; i < (I_DIM + 1) / 2; i++) W4[i] = (uint8_t)(xr() & 0xFF);
    printf("idot.h kernels tier=%s (I=%d, single core)\n", tier, I_DIM);

    double t;
    t = bench_i8(dot_i8i8);
    printf("  dot_i8i8          %8.1f ns  %6.2f GB/s(w+x)\n", t, I_DIM * 2.0 / t);
    t = bench_rows(dot_i8i8_rows, 3);
    printf("  dot_i8i8_rows n=3 %8.1f ns  %6.2f GB/s(w+3x)\n", t, I_DIM * 4.0 / t);
    t = bench_i4(dot_i4i8);
    printf("  dot_i4i8          %8.1f ns  %6.2f GB/s(w+x)\n", t, I_DIM * 1.5 / t);
    t = bench_qrow(qrow_i8);
    printf("  qrow_i8           %8.1f ns  %6.2f GB/s(x)\n", t, I_DIM * 4.0 / t);
    return 0;
}
