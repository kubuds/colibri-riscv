/* RVV micro-benchmark for the colibri integer-dot kernels on SpacemiT K3 (X100, VLEN=256).
 *
 * The X100 has RVV 1.0 + Zvbb/Zvfh/Zvfbfmin/Zvfbfwma but NO int8 dot-product
 * instruction (no Zvdot), so an exact int8xint8->int32 dot must be built from
 * widening multiplies. This benchmark picks the fastest exact formulation:
 *
 *   A) vwmul -> vwadd_wv accumulate chain, 1 or N independent accumulators,
 *      at LMUL m1 (32 elem/iter @VLEN256) or m2 (64 elem/iter)
 *   B) vwmul -> vredsum_vs per chunk (adds into acc lane 0)
 *   C) scalar (compiler-autovectorized reference)
 *
 * and for the packed int4xint8 dot, the nibble-assemble formulation:
 *   vzext.vf2 lane-local assemble ((t&0xF)|((t&0xF0)<<4)) then -8 per i8 lane,
 *   which places [b_jl-8, b_jh-8] at element positions 2j, 2j+1 exactly.
 *
 * Everything here is exact: integer sums are associative, and every variant is
 * checked bit-for-bit against the scalar reference before timing.
 */
#include <riscv_vector.h>
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

static int32_t ref_i8i8(const int8_t *w, const int8_t *x, int I) {
    int32_t s = 0;
    for (int i = 0; i < I; i++) s += (int32_t)w[i] * x[i];
    return s;
}
static int32_t ref_i4i8(const uint8_t *w4, const int8_t *x, int I) {
    int32_t s = 0;
    for (int i = 0; i < I; i++) {
        uint8_t b = w4[i >> 1];
        int v = (i & 1) ? ((int)(b >> 4) - 8) : ((int)(b & 0xF) - 8);
        s += v * x[i];
    }
    return s;
}

/* ---------------- variants: i8 x i8 ---------------- */

static inline int32_t dot_i8i8_m1_4acc(const int8_t *w, const int8_t *x, int I) {
    int i = 0;
    vint32m4_t a0 = __riscv_vmv_v_x_i32m4(0, __riscv_vsetvlmax_e32m4()), a1 = a0, a2 = a0, a3 = a0;
    int k = 0;
    for (; i < I; k++) {
        size_t vl = __riscv_vsetvl_e8m1(I - i);
        vint8m1_t  wv = __riscv_vle8_v_i8m1(w + i, vl);
        vint8m1_t  xv = __riscv_vle8_v_i8m1(x + i, vl);
        vint16m2_t p  = __riscv_vwmul_vv_i16m2(wv, xv, vl);
        switch (k & 3) {
        case 0: a0 = __riscv_vwadd_wv_i32m4_tu(a0, a0, p, vl); break;
        case 1: a1 = __riscv_vwadd_wv_i32m4_tu(a1, a1, p, vl); break;
        case 2: a2 = __riscv_vwadd_wv_i32m4_tu(a2, a2, p, vl); break;
        default: a3 = __riscv_vwadd_wv_i32m4_tu(a3, a3, p, vl); break;
        }
        i += vl;
    }
    size_t vm = __riscv_vsetvlmax_e32m4();
    vint32m1_t s1 = __riscv_vmv_v_x_i32m1(0, 1);
    a0 = __riscv_vadd_vv_i32m4(a0, a1, vm);
    a2 = __riscv_vadd_vv_i32m4(a2, a3, vm);
    a0 = __riscv_vadd_vv_i32m4(a0, a2, vm);
    s1 = __riscv_vredsum_vs_i32m4_i32m1(a0, s1, vm);
    return __riscv_vmv_x_s_i32m1_i32(s1);
}

static inline int32_t dot_i8i8_m1_1acc(const int8_t *w, const int8_t *x, int I) {
    int i = 0;
    vint32m4_t a0 = __riscv_vmv_v_x_i32m4(0, __riscv_vsetvlmax_e32m4());
    for (; i < I;) {
        size_t vl = __riscv_vsetvl_e8m1(I - i);
        vint8m1_t  wv = __riscv_vle8_v_i8m1(w + i, vl);
        vint8m1_t  xv = __riscv_vle8_v_i8m1(x + i, vl);
        vint16m2_t p  = __riscv_vwmul_vv_i16m2(wv, xv, vl);
        a0 = __riscv_vwadd_wv_i32m4_tu(a0, a0, p, vl);
        i += vl;
    }
    size_t vm = __riscv_vsetvlmax_e32m4();
    vint32m1_t s1 = __riscv_vmv_v_x_i32m1(0, 1);
    s1 = __riscv_vredsum_vs_i32m4_i32m1(a0, s1, vm);
    return __riscv_vmv_x_s_i32m1_i32(s1);
}

static inline int32_t dot_i8i8_m2_2acc(const int8_t *w, const int8_t *x, int I) {
    int i = 0;
    vint32m8_t a0 = __riscv_vmv_v_x_i32m8(0, __riscv_vsetvlmax_e32m8()), a1 = a0;
    int k = 0;
    for (; i < I; k++) {
        size_t vl = __riscv_vsetvl_e8m2(I - i);
        vint8m2_t  wv = __riscv_vle8_v_i8m2(w + i, vl);
        vint8m2_t  xv = __riscv_vle8_v_i8m2(x + i, vl);
        vint16m4_t p  = __riscv_vwmul_vv_i16m4(wv, xv, vl);
        if (k & 1) a1 = __riscv_vwadd_wv_i32m8_tu(a1, a1, p, vl);
        else       a0 = __riscv_vwadd_wv_i32m8_tu(a0, a0, p, vl);
        i += vl;
    }
    size_t vm = __riscv_vsetvlmax_e32m8();
    vint32m1_t s1 = __riscv_vmv_v_x_i32m1(0, 1);
    a0 = __riscv_vadd_vv_i32m8(a0, a1, vm);
    s1 = __riscv_vredsum_vs_i32m8_i32m1(a0, s1, vm);
    return __riscv_vmv_x_s_i32m1_i32(s1);
}

static inline int32_t dot_i8i8_m2_4acc(const int8_t *w, const int8_t *x, int I) {
    int i = 0;
    vint32m8_t a0 = __riscv_vmv_v_x_i32m8(0, __riscv_vsetvlmax_e32m8()), a1 = a0;
    vint32m8_t b0 = __riscv_vmv_v_x_i32m8(0, __riscv_vsetvlmax_e32m8()), b1 = b0;
    int k = 0;
    for (; i < I; k++) {
        size_t vl = __riscv_vsetvl_e8m2(I - i);
        vint8m2_t  wv = __riscv_vle8_v_i8m2(w + i, vl);
        vint8m2_t  xv = __riscv_vle8_v_i8m2(x + i, vl);
        vint16m4_t p  = __riscv_vwmul_vv_i16m4(wv, xv, vl);
        switch (k & 3) {
        case 0: a0 = __riscv_vwadd_wv_i32m8_tu(a0, a0, p, vl); break;
        case 1: a1 = __riscv_vwadd_wv_i32m8_tu(a1, a1, p, vl); break;
        case 2: b0 = __riscv_vwadd_wv_i32m8_tu(b0, b0, p, vl); break;
        default: b1 = __riscv_vwadd_wv_i32m8_tu(b1, b1, p, vl); break;
        }
        i += vl;
    }
    size_t vm = __riscv_vsetvlmax_e32m8();
    vint32m1_t s1 = __riscv_vmv_v_x_i32m1(0, 1);
    a0 = __riscv_vadd_vv_i32m8(a0, a1, vm);
    b0 = __riscv_vadd_vv_i32m8(b0, b1, vm);
    a0 = __riscv_vadd_vv_i32m8(a0, b0, vm);
    s1 = __riscv_vredsum_vs_i32m8_i32m1(a0, s1, vm);
    return __riscv_vmv_x_s_i32m1_i32(s1);
}

/* B) per-chunk vredsum into acc lane 0 */
static inline int32_t dot_i8i8_redsum(const int8_t *w, const int8_t *x, int I) {
    int i = 0;
    vint32m1_t acc = __riscv_vmv_v_x_i32m1(0, 1);
    for (; i < I;) {
        size_t vl = __riscv_vsetvl_e8m2(I - i);
        vint8m2_t  wv = __riscv_vle8_v_i8m2(w + i, vl);
        vint8m2_t  xv = __riscv_vle8_v_i8m2(x + i, vl);
        vint16m4_t p  = __riscv_vwmul_vv_i16m4(wv, xv, vl);
        acc = __riscv_vwredsum_vs_i16m4_i32m1(p, acc, vl);
        i += vl;
    }
    return __riscv_vmv_x_s_i32m1_i32(acc);
}

/* B2) two independent redsum chains (breaks the lane-0 dependency) */
static inline int32_t dot_i8i8_redsum2(const int8_t *w, const int8_t *x, int I) {
    int i = 0;
    vint32m1_t a0 = __riscv_vmv_v_x_i32m1(0, 1), a1 = a0;
    int k = 0;
    for (; i < I; k++) {
        size_t vl = __riscv_vsetvl_e8m2(I - i);
        vint8m2_t  wv = __riscv_vle8_v_i8m2(w + i, vl);
        vint8m2_t  xv = __riscv_vle8_v_i8m2(x + i, vl);
        vint16m4_t p  = __riscv_vwmul_vv_i16m4(wv, xv, vl);
        if (k & 1) a1 = __riscv_vwredsum_vs_i16m4_i32m1(p, a1, vl);
        else       a0 = __riscv_vwredsum_vs_i16m4_i32m1(p, a0, vl);
        i += vl;
    }
    return __riscv_vmv_x_s_i32m1_i32(a0) + __riscv_vmv_x_s_i32m1_i32(a1);
}

/* B3) LMUL m4 chunks (128 elem/iter @VLEN256) */
static inline int32_t dot_i8i8_redsum4(const int8_t *w, const int8_t *x, int I) {
    int i = 0;
    vint32m1_t acc = __riscv_vmv_v_x_i32m1(0, 1);
    for (; i < I;) {
        size_t vl = __riscv_vsetvl_e8m4(I - i);
        vint8m4_t  wv = __riscv_vle8_v_i8m4(w + i, vl);
        vint8m4_t  xv = __riscv_vle8_v_i8m4(x + i, vl);
        vint16m8_t p  = __riscv_vwmul_vv_i16m8(wv, xv, vl);
        acc = __riscv_vwredsum_vs_i16m8_i32m1(p, acc, vl);
        i += vl;
    }
    return __riscv_vmv_x_s_i32m1_i32(acc);
}

/* B4) redsum m2 but x loaded with widening mul pair: w even/odd merged via
 * vwadd on products of m1 pairs -- variant: m1 chunks */
static inline int32_t dot_i8i8_redsum_m1(const int8_t *w, const int8_t *x, int I) {
    int i = 0;
    vint32m1_t acc = __riscv_vmv_v_x_i32m1(0, 1);
    for (; i < I;) {
        size_t vl = __riscv_vsetvl_e8m1(I - i);
        vint8m1_t  wv = __riscv_vle8_v_i8m1(w + i, vl);
        vint8m1_t  xv = __riscv_vle8_v_i8m1(x + i, vl);
        vint16m2_t p  = __riscv_vwmul_vv_i16m2(wv, xv, vl);
        acc = __riscv_vwredsum_vs_i16m2_i32m1(p, acc, vl);
        i += vl;
    }
    return __riscv_vmv_x_s_i32m1_i32(acc);
}

/* B5) m4 chunks with 2 independent chains */
static inline int32_t dot_i8i8_redsum4_2(const int8_t *w, const int8_t *x, int I) {
    int i = 0;
    vint32m1_t a0 = __riscv_vmv_v_x_i32m1(0, 1), a1 = a0;
    int k = 0;
    for (; i < I; k++) {
        size_t vl = __riscv_vsetvl_e8m4(I - i);
        vint8m4_t  wv = __riscv_vle8_v_i8m4(w + i, vl);
        vint8m4_t  xv = __riscv_vle8_v_i8m4(x + i, vl);
        vint16m8_t p  = __riscv_vwmul_vv_i16m8(wv, xv, vl);
        if (k & 1) a1 = __riscv_vwredsum_vs_i16m8_i32m1(p, a1, vl);
        else       a0 = __riscv_vwredsum_vs_i16m8_i32m1(p, a0, vl);
        i += vl;
    }
    return __riscv_vmv_x_s_i32m1_i32(a0) + __riscv_vmv_x_s_i32m1_i32(a1);
}

static int32_t dot_i8i8_scalar(const int8_t *w, const int8_t *x, int I) {
    return ref_i8i8(w, x, I);
}

/* ---------------- variants: i4 x i8 ---------------- */

/* lane-local nibble assemble via vzext.vf2: byte j -> u16 lane j =
 * (b & 0xF) | ((b & 0xF0) << 4); the i8 view is then [b_l, b_h] per byte,
 * exactly the element order [e_{2j}, e_{2j+1}]. Subtract 8 per i8 lane.
 * The loop walks the even part of I (vsetvl keeps vl even when avl is), and
 * one scalar step closes an odd tail. */
static inline int32_t dot_i4i8_vzext(const uint8_t *w4, const int8_t *x, int I) {
    int32_t sum = 0;
    int even = I & ~1;
    int i = 0;
    vint32m8_t a0 = __riscv_vmv_v_x_i32m8(0, __riscv_vsetvlmax_e32m8()), a1 = a0;
    int k = 0;
    for (; i < even; k++) {
        size_t vl = __riscv_vsetvl_e8m2(even - i);   /* even: vlmax(e8m2) and even-i are even */
        size_t nb = vl / 2;
        vuint8m1_t  b8 = __riscv_vle8_v_u8m1(w4 + (i >> 1), nb);
        vuint16m2_t t  = __riscv_vzext_vf2_u16m2(b8, nb);
        vuint16m2_t n  = __riscv_vor_vv_u16m2(
            __riscv_vand_vx_u16m2(t, 0x0F, nb),
            __riscv_vsll_vx_u16m2(__riscv_vand_vx_u16m2(t, 0xF0, nb), 4, nb), nb);
        vint8m2_t wv = __riscv_vsub_vx_i8m2(__riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u16m2_u8m2(n)), 8, vl);
        vint8m2_t xv = __riscv_vle8_v_i8m2(x + i, vl);
        vint16m4_t p = __riscv_vwmul_vv_i16m4(wv, xv, vl);
        if (k & 1) a1 = __riscv_vwadd_wv_i32m8_tu(a1, a1, p, vl);
        else       a0 = __riscv_vwadd_wv_i32m8_tu(a0, a0, p, vl);
        i += vl;
    }
    if (i < I) {
        uint8_t b = w4[i >> 1];
        sum += ((int)(b & 0xF) - 8) * x[i];
    }
    size_t vm = __riscv_vsetvlmax_e32m8();
    vint32m1_t s1 = __riscv_vmv_v_x_i32m1(0, 1);
    a0 = __riscv_vadd_vv_i32m8(a0, a1, vm);
    s1 = __riscv_vredsum_vs_i32m8_i32m1(a0, s1, vm);
    return sum + __riscv_vmv_x_s_i32m1_i32(s1);
}

/* same assemble at m1 (32 elem/iter), 4 accumulators */
static inline int32_t dot_i4i8_vzext_m1(const uint8_t *w4, const int8_t *x, int I) {
    int32_t sum = 0;
    int even = I & ~1;
    int i = 0;
    vint32m4_t a0 = __riscv_vmv_v_x_i32m4(0, __riscv_vsetvlmax_e32m4()), a1 = a0, a2 = a0, a3 = a0;
    int k = 0;
    for (; i < even; k++) {
        size_t vl = __riscv_vsetvl_e8m1(even - i);
        size_t nb = vl / 2;
        vuint8mf2_t b8 = __riscv_vle8_v_u8mf2(w4 + (i >> 1), nb);
        vuint16m1_t t  = __riscv_vzext_vf2_u16m1(b8, nb);
        vuint16m1_t n  = __riscv_vor_vv_u16m1(
            __riscv_vand_vx_u16m1(t, 0x0F, nb),
            __riscv_vsll_vx_u16m1(__riscv_vand_vx_u16m1(t, 0xF0, nb), 4, nb), nb);
        vint8m1_t wv = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(__riscv_vreinterpret_v_u16m1_u8m1(n)), 8, vl);
        vint8m1_t xv = __riscv_vle8_v_i8m1(x + i, vl);
        vint16m2_t p = __riscv_vwmul_vv_i16m2(wv, xv, vl);
        switch (k & 3) {
        case 0: a0 = __riscv_vwadd_wv_i32m4_tu(a0, a0, p, vl); break;
        case 1: a1 = __riscv_vwadd_wv_i32m4_tu(a1, a1, p, vl); break;
        case 2: a2 = __riscv_vwadd_wv_i32m4_tu(a2, a2, p, vl); break;
        default: a3 = __riscv_vwadd_wv_i32m4_tu(a3, a3, p, vl); break;
        }
        i += vl;
    }
    if (i < I) {
        uint8_t b = w4[i >> 1];
        sum += ((int)(b & 0xF) - 8) * x[i];
    }
    size_t vm = __riscv_vsetvlmax_e32m4();
    vint32m1_t s1 = __riscv_vmv_v_x_i32m1(0, 1);
    a0 = __riscv_vadd_vv_i32m4(a0, a1, vm);
    a2 = __riscv_vadd_vv_i32m4(a2, a3, vm);
    a0 = __riscv_vadd_vv_i32m4(a0, a2, vm);
    s1 = __riscv_vredsum_vs_i32m4_i32m1(a0, s1, vm);
    return sum + __riscv_vmv_x_s_i32m1_i32(s1);
}

/* redsum variant for i4 */
static inline int32_t dot_i4i8_vzext_red(const uint8_t *w4, const int8_t *x, int I) {
    int32_t sum = 0;
    int even = I & ~1;
    int i = 0;
    vint32m1_t a0 = __riscv_vmv_v_x_i32m1(0, 1), a1 = a0;
    int k = 0;
    for (; i < even; k++) {
        size_t vl = __riscv_vsetvl_e8m2(even - i);
        size_t nb = vl / 2;
        vuint8m1_t  b8 = __riscv_vle8_v_u8m1(w4 + (i >> 1), nb);
        vuint16m2_t t  = __riscv_vzext_vf2_u16m2(b8, nb);
        vuint16m2_t n  = __riscv_vor_vv_u16m2(
            __riscv_vand_vx_u16m2(t, 0x0F, nb),
            __riscv_vsll_vx_u16m2(__riscv_vand_vx_u16m2(t, 0xF0, nb), 4, nb), nb);
        vint8m2_t wv = __riscv_vsub_vx_i8m2(__riscv_vreinterpret_v_u8m2_i8m2(__riscv_vreinterpret_v_u16m2_u8m2(n)), 8, vl);
        vint8m2_t xv = __riscv_vle8_v_i8m2(x + i, vl);
        vint16m4_t p = __riscv_vwmul_vv_i16m4(wv, xv, vl);
        if (k & 1) a1 = __riscv_vwredsum_vs_i16m4_i32m1(p, a1, vl);
        else       a0 = __riscv_vwredsum_vs_i16m4_i32m1(p, a0, vl);
        i += vl;
    }
    if (i < I) {
        uint8_t b = w4[i >> 1];
        sum += ((int)(b & 0xF) - 8) * x[i];
    }
    return sum + __riscv_vmv_x_s_i32m1_i32(a0) + __riscv_vmv_x_s_i32m1_i32(a1);
}

static int32_t dot_i4i8_scalar(const uint8_t *w4, const int8_t *x, int I) {
    return ref_i4i8(w4, x, I);
}

/* ---------------- bench harness ---------------- */

#define I_DIM 4096
#define N_REPEAT 20000

static int8_t  W[I_DIM + 64] __attribute__((aligned(64)));
static int8_t  X[I_DIM + 64] __attribute__((aligned(64)));
static uint8_t W4[I_DIM / 2 + 64] __attribute__((aligned(64)));

typedef int32_t (*fn_i8)(const int8_t *, const int8_t *, int);
typedef int32_t (*fn_i4)(const uint8_t *, const int8_t *, int);

static void bench_i8(const char *name, fn_i8 f) {
    static const int sizes[] = {1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 4096, 4097};
    for (unsigned t = 0; t < sizeof(sizes) / sizeof(sizes[0]); t++) {
        int I = sizes[t];
        for (int i = 0; i < I; i++) X[i] = (int8_t)((int)(xr() % 255) - 127);
        for (int i = 0; i < I; i++) W[i] = (int8_t)((int)(xr() % 256) - 128);
        if (f(W, X, I) != ref_i8i8(W, X, I)) {
            printf("FAIL %s I=%d: %d != %d\n", name, I, f(W, X, I), ref_i8i8(W, X, I));
            exit(1);
        }
    }
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
    double ns_per = (double)best / N_REPEAT;
    double gbps = (double)I * 2.0 / ns_per;
    printf("%-22s %8.1f ns/call  %6.2f GB/s(w+x)   [checksum %d]\n", name, ns_per, gbps, sink);
}

static void bench_i4(const char *name, fn_i4 f) {
    static const int sizes[] = {1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 4096, 4097};
    for (unsigned t = 0; t < sizeof(sizes) / sizeof(sizes[0]); t++) {
        int I = sizes[t];
        for (int i = 0; i < I; i++) X[i] = (int8_t)((int)(xr() % 255) - 127);
        for (int i = 0; i < (I + 1) / 2; i++) W4[i] = (uint8_t)(xr() & 0xFF);
        if (f(W4, X, I) != ref_i4i8(W4, X, I)) {
            printf("FAIL %s I=%d: %d != %d\n", name, I, f(W4, X, I), ref_i4i8(W4, X, I));
            exit(1);
        }
    }
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
    double ns_per = (double)best / N_REPEAT;
    double gbps = ((double)I * 1.5) / ns_per;
    printf("%-22s %8.1f ns/call  %6.2f GB/s(w+x)   [checksum %d]\n", name, ns_per, gbps, sink);
}

int main(void) {
    printf("K3 X100 RVV idot micro-benchmark (VLEN=%zu)\n", (size_t)__riscv_vlenb() * 8);
    printf("-- dot_i8i8 (I=%d, %d reps) --\n", I_DIM, N_REPEAT);
    bench_i8("scalar", dot_i8i8_scalar);
    bench_i8("m1_4acc", dot_i8i8_m1_4acc);
    bench_i8("m1_1acc", dot_i8i8_m1_1acc);
    bench_i8("m2_2acc", dot_i8i8_m2_2acc);
    bench_i8("m2_4acc", dot_i8i8_m2_4acc);
    bench_i8("redsum chunk", dot_i8i8_redsum);
    bench_i8("redsum2 chunk", dot_i8i8_redsum2);
    bench_i8("redsum m4 chunk", dot_i8i8_redsum4);
    bench_i8("redsum4 2chain", dot_i8i8_redsum4_2);
    bench_i8("redsum m1 chunk", dot_i8i8_redsum_m1);
    printf("-- dot_i4i8 (I=%d, %d reps) --\n", I_DIM, N_REPEAT);
    bench_i4("scalar", dot_i4i8_scalar);
    bench_i4("vzext m2_2acc", dot_i4i8_vzext);
    bench_i4("vzext m1_4acc", dot_i4i8_vzext_m1);
    bench_i4("vzext redsum2", dot_i4i8_vzext_red);
    return 0;
}
