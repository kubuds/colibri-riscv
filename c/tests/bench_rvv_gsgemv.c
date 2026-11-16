/* matmul_q_gs (gsgemv.h) throughput on SpacemiT K3, same source built twice:
 *   -mcpu=spacemit-x100  -> the RVV tier
 *   -march=rv64gc        -> the scalar fallback
 * Prints ns/output-row for the two expert shapes qwen36 actually runs
 * (I=2048,O=512 gs=64 gate/up and I=512,O=2048 gs=64 down), single core.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../gsgemv.h"

static uint64_t ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static uint32_t rs = 0x1234u;
static uint32_t xr(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }

#define I_MAX 2048
#define O_MAX 2048

static float   X[I_MAX];
static int8_t  Q[(size_t)I_MAX * O_MAX];
static float   SC[(size_t)O_MAX * (I_MAX / 64 + 1)];
static float   Y[O_MAX];

int main(void) {
#if defined(__riscv_vector) && !defined(COLI_NO_RVV)
    const char *tier = "rvv";
#elif defined(__AVX2__)
    const char *tier = "avx2";
#else
    const char *tier = "scalar";
#endif
    for (int i = 0; i < I_MAX; i++) X[i] = ((float)(xr() % 2001) - 1000.f) / 500.f;
    for (size_t i = 0; i < (size_t)I_MAX * O_MAX; i++) Q[i] = (int8_t)(xr() & 0xFF);
    for (int i = 0; i < O_MAX * (I_MAX / 64 + 1); i++) SC[i] = ((float)(xr() % 1000) - 500.f) / 100000.f;

    struct { const char *name; int I, O, gs; } shapes[] = {
        {"gate/up  I=2048 O=512  gs=64", 2048, 512, 64},
        {"down     I=512  O=2048 gs=64", 512, 2048, 64},
        {"ragged   I=2000 O=512  gs=64", 2000, 512, 64},
    };
    printf("matmul_q_gs tier=%s\n", tier);
    for (unsigned s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        int I = shapes[s].I, O = shapes[s].O, gs = shapes[s].gs;
        int reps = 20;
        volatile float sink = 0;
        uint64_t best = ~0ull;
        for (int rep = 0; rep < 3; rep++) {
            uint64_t t0 = ns_now();
            for (int r = 0; r < reps; r++) matmul_q_gs(Y, X, Q, SC, I, O, gs);
            uint64_t t1 = ns_now();
            if (t1 - t0 < best) best = t1 - t0;
        }
        sink += Y[0];
        double ns_row = (double)best / (reps * (double)O);
        double gflops = (double)I * 2.0 / ns_row;
        printf("%-32s %9.1f ns/row  %6.2f GFLOP-ish/s/core  [%.3f]\n",
               shapes[s].name, ns_row, gflops, (double)sink);
    }
    return 0;
}
