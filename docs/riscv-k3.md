# colibri on RISC-V: the SpacemiT K3 port

This document records the RISC-V port of colibri, targeting the SpacemiT **K3**
SoC (8× X100 general-purpose cores + 8× A100 AI cores, 12 nm, 30 W), and the
optimization work carried over from SpacemiT's
[ggml-spacemit](https://github.com/spacemit-com/llama.cpp/tree/mtmd-backend/ggml/src/ggml-spacemit)
backend. Every claim in here was measured on real K3 hardware
(`k3pico16g.local`, 8 schedulable X100 cores, 16 GB RAM, Bianbu 4.0.8,
gcc 15.2), not simulated.

## 1. The machine

The K3 compute subsystem (from the SpacemiT K3 paper, "SpacemiT K3: A RVA23
RISC-V AI CPU with 60 TOPS AI Compute"):

| component | detail |
|---|---|
| X100 core | RVA23 profile, 4-wide out-of-order, derived from OpenC910 (Xuantie C910) |
| X100 vector unit | **RVV 1.0, VLEN=256, ELEN=64**; 2× 128-bit arithmetic pipes + 2× 128-bit load/store pipes (an LMUL=1 op issues as two 128-bit µops) |
| X100 ISA extras | Zvbb, Zvfh, Zfbfmin, Zvfbfmin, Zvfbfwma, Zvkt, vector crypto |
| A100 core | AI-CPU with SpacemiT-IME instruction extensions (60 TOPS INT4 dense / 30 TOPS INT8 / 7.5 TOPS BF16-FP16-FP8) |
| memory | 2× 32-bit LPDDR5 controllers, 256-bit internal bus |

Platform facts the port had to learn (each bit us was found the hard way,
on the machine):

* **16 harts online, 8 schedulable.** `getconf _NPROCESSORS_ONLN` and
  `sysconf(_SC_NPROCESSORS_ONLN)` report 16 (X100 + A100), but an ordinary
  process may only run on the 8 X100 cores: `sched_setaffinity` with any
  A100 hart in the mask fails with `EINVAL`. Every "widen to all CPUs"
  code path (OMP hot-team re-exec in `omp_tune.h`/`colibri.c`, the CUDA
  uploader widen in `qwen36_tier.c`) built its mask as `0..N-1` and failed
  silently on K3. They now grow the mask one CPU at a time and keep what
  sticks, which is correct on symmetric and asymmetric SoCs alike.
* gcc on Bianbu knows the core: `gcc -mcpu=spacemit-x100` enables RVV with
  `__riscv_v_min_vlen 256` plus Zvbb/Zvfh/Zvfbfmin/Zvfbfwma. Neither
  `-march=native` nor `-mcpu=native` works on this toolchain — the Makefile
  uses `-mcpu=$(ARCH)` with `ARCH ?= spacemit-x100` and `ARCH=generic` as
  the no-V scalar baseline.

## 2. Build

`c/Makefile` grew a `riscv64` branch (detected from `-dumpmachine` like the
other platforms):

```console
$ make colibri            # tuned for the X100 (-mcpu=spacemit-x100)
$ make test-c             # the C test suite
$ make colibri ARCH=generic   # scalar-only rv64gc baseline
```

Everything compiled and ran through its scalar fallbacks with no source
changes on the first pass — the engine's tier structure (`__AVX2__` →
`__SSE4_1__` → scalar, `__ARM_NEON` → scalar) already degrades cleanly.
The work below adds the `__riscv_vector` tiers where the scalar fallback
was the hot path.

## 3. RVV kernels, and what ggml-spacemit taught

The X100 has **no integer dot-product instruction** (no Zvdot). ggml-spacemit
deals with this by repacking quantized weights into blocked layouts and
computing GEMMs in **f16** (Zvfh on the two vector pipes), or by running the
IME custom instructions on the A100 cores. colibri's integer kernels have a
stricter contract — the integer dots are **bit-exact** against a plain-C
reference by design (`tests/test_idot.c`) — so they cannot be moved to f16.
The port therefore builds the exact dots from widening integer ops and takes
the f16/convert lessons where the contract allows.

### 3.1 `idot.h` — the exact integer dots

`dot_i8i8` / `dot_i8i8_rows` / `dot_i4i8` are the workhorses of every engine
(int8- or int4-quantized weights × int8-quantized activations). The RVV tier:

* widen-mul `i8 × i8 → i16` (`vwmul`; `|w·x| ≤ 16256`, never overflows),
* widen-reduce `i16 → i32` per chunk (`vwredsum`, seeds a single lane),
* two independent reduction chains per dot to break the lane-0 dependency.

Measured on one X100 core, I = 4096 (`bench/rvv_idot_micro.c`, exactness
checked bit-for-bit against the scalar reference at 17 sizes before timing):

| kernel | scalar | RVV | speedup |
|---|---|---|---|
| `dot_i8i8` | 728 ns | **334 ns** | 2.2× |
| `dot_i4i8` (packed int4) | 5533 ns | **881 ns** | 6.3× |

The micro-benchmark rejected the "obvious" shapes: `vwadd` accumulator chains
were *slower* than scalar (uop-heavy widening adds; 4 accumulators helped a
little, LMUL=8 spilled), while the per-chunk `vwredsum` formulation won by
keeping the cross-chunk dependency on one scalar-sized seed. That is the
same "break the dependency chain, measure the shape" discipline
ggml-spacemit's kernels document throughout.

`dot_i4i8`'s nibble unpack uses a lane-local trick: zero-extend the packed
byte to u16 and compute `(t & 0x0F) | ((t & 0xF0) << 4)` — byte `j`'s low
nibble lands at bits 0..3 and its high nibble at bits 8..11 of the same
16-bit lane, so the i8 view is exactly the element order `[e_2j, e_2j+1]`
with no cross-lane shuffle. One per-lane i8 `vsub` folds in the nibble
offset. (This is the vector analogue of the `unpacklo/hi` pairs the x86
kernels pay for.)

`qrow_i8` / `dense_act_i8` (activation quantization): `vfredmax` for the
amax, `vfncvt` (round-to-nearest-even, same as `lrintf`) + `vncvt` for the
quantize, `vwredsum` for the 64-element block sums.

### 3.2 `st.h` — BF16/F16 → F32 bulk conversion

Every shard read and every streamed expert passes through these. The BF16
conversion is the same exact zero-pad widening as every other tier (shift,
no arithmetic). For F16 the hardware `vfwcvt` (Zvfhmin, mandatory in RVA23)
is exact for normals, subnormals and inf — but **canonicalizes NaNs**,
while the scalar reference preserves `sign | 0x7F800000 | (man << 13)`.
The tier patches exactly those lanes with a `vmerge` (exhaustively checked:
`tests/test_st_f16_bf16_simd` verifies all 65536 patterns bit-for-bit on
the K3). This is a place where a naive "hardware convert is IEEE, ship it"
would have leaked 2045 wrong bit patterns.

### 3.3 `gsgemv.h` — qwen36's hottest kernel

Group-scaled int8 GEMV (`matmul_q_gs`): four output rows in flight, two
chunk accumulators per row, one pinned `fmaf` per group — the SSE4.1 tier's
shape at full vector width, with `vsext.vf4` + `vfcvt` dequant and `vfmacc`
against the f32 activation. Unlike the AVX2/SSE4.1 fixed-width loops (whose
sub-chunk tail-drop is pinned as behavior by `tests/test_gsgemv.c` P5), the
RVV tier carries every tail through `vsetvl`. Like the SSE4.1 tier it is a
new reduction shape (`vfredusum` is unordered) and is checked by tolerance.

One X100 core, the two expert shapes qwen36 runs (`bench_rvv_gsgemv`,
scalar build vs RVV build of the same source):

| shape | scalar | RVV | speedup |
|---|---|---|---|
| gate/up I=2048 O=512 gs=64 | 3279 ns/row | **1114 ns/row** | 2.9× |
| down I=512 O=2048 gs=64 | 822 ns/row | **257 ns/row** | 3.2× |
| ragged I=2000 O=512 gs=64 | 3188 ns/row | **987 ns/row** | 3.2× |

### 3.4 `quant.h` / `olmoe.c` — the default f32 GEMV paths

`matmul_q` (int8 rows) and `matmul_i4` / `matmul_i4_pair` (packed int4
pairs) are the engines' *default* dequant-on-use f32 GEMVs (the integer
IDOT paths are opt-in via `IDOT=1`). They got the same RVV treatment:
weights widened to f32 (`vsext.vf4` + `vfcvt`; the int4 pair uses the
lane-local nibble assemble), `vfmacc` with two chunk accumulators,
`vsetvl` tails. `olmoe.c`'s exact 16-element `dot_i8_16` got an RVV
widen-mul/widen-reduce tier so the `IDOT=1` path is reachable on RISC-V
(`tests/test_int_kernel_exact`: 10544 checks, 0 failures on the K3).
The float reduction shapes are checked with tolerance
(`tests/test_olmoe_matmul_q.c`), like the NEON tier before them.

### 3.5 `quant.h` / `colibri.c` — contraction pinning (portability fix)

`matmul_i4_grouped` (quant.h) and `matmul_i4_grouped_pair` (colibri.c) are
fused/unfused twins that must agree **bitwise** (`tests/test_i4_grouped`).
On gcc-15/riscv64 they drifted 1 ulp: `-ffp-contract=fast` lets the compiler
fuse `x*y + z` per *code shape*, and the twins' shapes (one with a redundant
inner branch, one SLP-vectorized across its two outputs) got different
fusions. `fmaf()` pinning fixed the outer accumulate but exposed the same
ambiguity in the inner product-sum. The fix pins **every rounding point**
with `coli_fp_pin`, an empty `asm volatile("" : "+f"(x))` barrier: zero
instructions, but the multiply, the pair-sum and the accumulate each round
exactly once — the same non-fused canonical the x86 builds already produced
(and the same convention as sse41_kernels.h's explicit `mul+add` pinning),
so no x86 bits move.

## 4. Validation on the K3

Everything below runs on the K3 itself:

* `make test-c` — the full C suite (233 test binaries), including the
  bit-exactness gates: `test_idot` (kernel + driver exactness on the RVV
  path), `test_st_f16_bf16_simd` (exhaustive 65536-pattern conversion
  sweep), `test_i4_grouped` (fused-vs-unfused twins), `test_gsgemv`,
  `test_qwen36_tier_fp8` (uploader affinity widen), and the rest.
* `make test-python` — the tiny-model end-to-end oracles (generated
  fixtures + token-exact engine runs).
* `tests/bench_rvv_idot`, `tests/bench_rvv_gsgemv` — the real-kernel
  micro-benchmarks quoted above (build twice, `ARCH=generic` for the
  scalar tier).
* `tests/bench_rvv_idot_shapes` — the design-shape micro-benchmark that
  picked the widen-mul + `vwredsum` formulation over the `vwadd`
  accumulator chains and the per-chunk redsum over LMUL variants.

## 5. Not done (and why)

* **A100 / IME.** The A100 AI cores and their SpacemiT-IME instructions are
  reached through SpacemiT's spine-runtime (`libspert`), which owns core
  assignment, the shared TCM buffers and the barrier machinery —
  ggml-spacemit's `ime.cpp`/`ime2_kernels.cpp` are built on it. colibri is
  a zero-dependency pure-C engine by charter; wiring it to a vendor runtime
  is a separate architectural decision. The RVV work above runs on the
  X100 cores every process can use.
* **ggml-spacemit's repacked f16 GEMM path.** Its q4→f16 32×K32 repack +
  `vfmacc` tiles are the right idea where float results are tolerance-checked
  anyway; colibri's integer paths are contractually bit-exact and stay
  integer. The f16 lesson is already applied where the contract allows
  (the `st.h` conversions).
