// Volta (sm_70) chunked Gated DeltaNet: the same three stages, workspace and equations as the
// sm_80 pipeline (prepare_wy_wu / state_passing / output, see chunked/launch.cu), written for
// the tensor cores Volta has: WMMA m16n16k16 with FP16 operands and FP32 accumulation.
//
// Per value head h and 64-token chunk c (G = inclusive in-chunk cumsum of g, b = beta):
//   prepare   N[i][j] = -b_i (k_i.k_j) exp(G_i - G_j) for i > j, else 0
//             T = (I - N)^{-1}               (FP32 forward substitution, in place)
//             W = T (b exp(G) . K),  U = T (b . V)                         -> BF16 workspace
//   state     v_new = U - W h;  inter = exp(G_t) q_t h  (h = state at chunk start)
//             h = exp(G_63) h + K^T (exp(G_63 - G_i) . v_new)            (FP32 state)
//   output    o_t = scale (inter_t + sum_{s<=t} (q_t.k_s) exp(G_t - G_s) v_new_s)
//
// Differences from the sm_80 kernels, all private: T is inverted in FP32 SIMT (16x16 diagonal
// blocks by forward substitution, then the off-diagonal blocks column by column) instead of TF32
// block Schur; W, U, v_new and inter hold FP16 (natural column order) in BF16-typed
// workspace regions, plus FP16 copies of q and k (Volta-only regions), so the WMMA operands load
// straight from global memory and the stages need 25-50 KiB of shared memory; every exp is the
// accurate expf. Operands the sm_80 path feeds to TF32 MMAs are FP16 here (same 10-bit mantissa,
// narrower exponent range). Layouts are those of chunked::compute_workspace_layout().

#include "ops/linear_attention/gated_delta_net/launch.h"

#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/chunked/launch.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>

namespace ninfer::ops::detail::gated_delta_net {
namespace {

using namespace nvcuda;

constexpr int kD       = kStateDim;  // 128
constexpr int kBT      = kChunkSize; // 64
constexpr int kThreads = 128;        // 4 warps
constexpr int kHalfLd  = kD + 8;     // 136: FP16 rows of 128, padded against bank conflicts
constexpr int kTileLd  = kBT + 8;    // 72: FP16 rows of 64
constexpr int kF32Ld64 = kBT + 4;    // 68: FP32 rows of 64

using FragA    = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;
using FragAT   = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major>;
using FragB    = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>;
using FragBT   = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>;
using FragAcc  = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

// 64 rows x 128 BF16 values (row stride `row_stride` elements in global) -> FP16 smem rows,
// optionally scaled per row.
__device__ __forceinline__ void load_rows_bf16(const __nv_bfloat16* __restrict__ src,
                                               std::int64_t row_stride, __half (*dst)[kHalfLd],
                                               const float* row_scale) {
    for (int idx = static_cast<int>(threadIdx.x); idx < kBT * (kD / 8); idx += kThreads) {
        const int row = idx / (kD / 8);
        const int col = (idx % (kD / 8)) * 8;
        const uint4 raw = *reinterpret_cast<const uint4*>(src + row * row_stride + col);
        const auto* b   = reinterpret_cast<const __nv_bfloat16*>(&raw);
        const float s   = row_scale != nullptr ? row_scale[row] : 1.0f;
        __align__(16) __half h[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) { h[e] = __float2half_rn(__bfloat162float(b[e]) * s); }
        *reinterpret_cast<uint4*>(&dst[row][col]) = *reinterpret_cast<const uint4*>(h);
    }
}

// --------------------------------------------------------------------------------------------
// Stage 1: g cumsum, T = (I - N)^{-1}, W, U (+ FP16 copies of q and k).  Grid (chunks, H_v).
// --------------------------------------------------------------------------------------------
constexpr int kPrepThreads = 256; // 8 warps
constexpr int kBlk         = 16;  // T is inverted in 16x16 blocks

struct PrepareSmem {
    __half k[kBT][kHalfLd];  // K, later b exp(G) K, later b V
    union {
        float m[kBT][kF32Ld64]; // Gram -> N -> T (strictly lower part, unit diagonal implicit)
        __half t[kBT][kTileLd]; // then T in FP16 (unit diagonal explicit)
    };
    float stage[8][16 * 16]; // per-warp accumulator staging; block products during the inverse
    float g[kBT];
    float b[kBT];
    float scale[kBT];
};

__device__ __forceinline__ void store_tile_f16(const FragAcc& acc, float* stage,
                                               __half* __restrict__ dst, std::int64_t row_stride) {
    wmma::store_matrix_sync(stage, acc, 16, wmma::mem_row_major);
    __syncwarp();
    const int lane = static_cast<int>(threadIdx.x) & 31;
#pragma unroll
    for (int e = lane; e < 256; e += 32) {
        dst[(e / 16) * row_stride + (e % 16)] = __float2half_rn(stage[e]);
    }
    __syncwarp();
}

// out[64][128] = T[64][64] @ X[64][128] (X in s.k), FP16 rows of `out`.
__device__ __forceinline__ void t_times_rows(PrepareSmem& s, __half* __restrict__ out,
                                             std::int64_t row_stride) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    for (int tile = warp; tile < (kBT / 16) * (kD / 16); tile += kPrepThreads / 32) {
        const int ti = tile / (kD / 16);
        const int tj = tile % (kD / 16);
        FragAcc acc;
        wmma::fill_fragment(acc, 0.0f);
        for (int kk = 0; kk <= ti; ++kk) { // T is lower triangular
            FragA a;
            FragB b;
            wmma::load_matrix_sync(a, &s.t[ti * 16][kk * 16], kTileLd);
            wmma::load_matrix_sync(b, &s.k[kk * 16][tj * 16], kHalfLd);
            wmma::mma_sync(acc, a, b, acc);
        }
        store_tile_f16(acc, s.stage[warp], out + (ti * 16) * row_stride + tj * 16, row_stride);
    }
}

// Strictly-lower entry (r, c) of block row/col (bi, bj) of T, with the implicit unit diagonal.
__device__ __forceinline__ float t_entry(const PrepareSmem& s, int row, int col) {
    return row == col ? 1.0f : (row > col ? s.m[row][col] : 0.0f);
}

__global__ __launch_bounds__(kPrepThreads, 2) void gdn_volta_prepare_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const float* __restrict__ g_in,
    const float* __restrict__ beta, __half* __restrict__ W, __half* __restrict__ U,
    float* __restrict__ g_cumsum, __half* __restrict__ q16, __half* __restrict__ k16, int H_qk,
    int H_v) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    PrepareSmem& s = *reinterpret_cast<PrepareSmem*>(smem_raw);
    const int tid   = static_cast<int>(threadIdx.x);
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const int c     = static_cast<int>(blockIdx.x);
    const int h     = static_cast<int>(blockIdx.y);
    const int group = H_v / H_qk;
    const int qh    = h / group;
    const std::int64_t t0     = static_cast<std::int64_t>(c) * kBT;
    const std::int64_t row_qk = static_cast<std::int64_t>(H_qk) * kD;
    const std::int64_t row_v  = static_cast<std::int64_t>(H_v) * kD;

    if (tid < kBT) {
        s.g[tid] = g_in[(t0 + tid) * H_v + h];
        s.b[tid] = beta[(t0 + tid) * H_v + h];
    }
    load_rows_bf16(k + (t0 * H_qk + qh) * kD, row_qk, s.k, nullptr);
    __syncthreads();
    if (h % group == 0) { // one value head per Q/K head publishes the FP16 copies
        for (int idx = tid; idx < kBT * (kD / 8); idx += kPrepThreads) {
            const int row = idx / (kD / 8);
            const int col = (idx % (kD / 8)) * 8;
            const std::int64_t at = (t0 + row) * row_qk + qh * kD + col;
            *reinterpret_cast<uint4*>(k16 + at) = *reinterpret_cast<const uint4*>(&s.k[row][col]);
            const uint4 raw = *reinterpret_cast<const uint4*>(q + at);
            const auto* bq  = reinterpret_cast<const __nv_bfloat16*>(&raw);
            __align__(16) __half x[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) { x[e] = __float2half_rn(__bfloat162float(bq[e])); }
            *reinterpret_cast<uint4*>(q16 + at) = *reinterpret_cast<const uint4*>(x);
        }
    }
    if (tid < kBT) { // inclusive cumsum, summed in token order
        float acc = 0.0f;
        for (int j = 0; j <= tid; ++j) { acc += s.g[j]; }
        s.scale[tid] = acc;
    }
    __syncthreads();
    if (tid < kBT) {
        s.g[tid] = s.scale[tid];
        g_cumsum[(t0 + tid) * H_v + h] = s.scale[tid];
        s.scale[tid] = s.b[tid] * expf(s.scale[tid]); // b_i exp(G_i), for W
    }

    // Gram K K^T on the lower block triangle (10 of 16 tiles).
    for (int tile = warp; tile < 10; tile += kPrepThreads / 32) {
        int ti = 0;
        while ((ti + 1) * (ti + 2) / 2 <= tile) { ++ti; }
        const int tj = tile - ti * (ti + 1) / 2;
        FragAcc acc;
        wmma::fill_fragment(acc, 0.0f);
#pragma unroll
        for (int kk = 0; kk < kD / 16; ++kk) {
            FragA a;
            FragBT b;
            wmma::load_matrix_sync(a, &s.k[ti * 16][kk * 16], kHalfLd);
            wmma::load_matrix_sync(b, &s.k[tj * 16][kk * 16], kHalfLd);
            wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(&s.m[ti * 16][tj * 16], acc, kF32Ld64, wmma::mem_row_major);
    }
    __syncthreads();

    // N, strictly lower. A select, not a mask multiply: exp(G_i - G_j) for i < j may be +inf.
    for (int idx = tid; idx < kBT * kBT; idx += kPrepThreads) {
        const int i = idx / kBT;
        const int j = idx % kBT;
        s.m[i][j] = i > j ? -s.b[i] * s.m[i][j] * expf(s.g[i] - s.g[j]) : 0.0f;
    }
    __syncthreads();

    // T = (I - N)^{-1} by 16x16 blocks, in place over the strict lower triangle.
    // Diagonal blocks: forward substitution, one warp each (row r still holds N while solved).
    if (warp < kBT / kBlk) {
        const int o = warp * kBlk;
        for (int r = 1; r < kBlk; ++r) {
            float val = 0.0f;
            if (lane < r) {
                val = s.m[o + r][o + lane];
                for (int j = lane + 1; j < r; ++j) {
                    val = fmaf(s.m[o + r][o + j], s.m[o + j][o + lane], val);
                }
            }
            __syncwarp();
            if (lane < r) { s.m[o + r][o + lane] = val; }
            __syncwarp();
        }
    }
    __syncthreads();
    // Off-diagonal blocks, right to left by block column J (T L = I with L = I - N):
    //   T_IJ = (sum_{k=J+1..I} T_Ik N_kJ) T_JJ.
    // Column J's N blocks are read before any of them is overwritten (two barriers).
    float* P = &s.stage[0][0]; // up to 3 blocks of 16x16
    for (int J = kBT / kBlk - 2; J >= 0; --J) {
        const int nb = kBT / kBlk - 1 - J; // blocks I = J+1 .. 3
        for (int e = tid; e < nb * kBlk * kBlk; e += kPrepThreads) {
            const int I = J + 1 + e / (kBlk * kBlk);
            const int r = (e / kBlk) % kBlk;
            const int x = e % kBlk;
            float p = 0.0f;
            for (int kb = J + 1; kb <= I; ++kb) {
                for (int mm = 0; mm < kBlk; ++mm) {
                    p = fmaf(t_entry(s, I * kBlk + r, kb * kBlk + mm),
                             s.m[kb * kBlk + mm][J * kBlk + x], p);
                }
            }
            P[e] = p;
        }
        __syncthreads();
        float out[3];
        for (int e = tid, n = 0; e < nb * kBlk * kBlk; e += kPrepThreads, ++n) {
            const int r   = (e / kBlk) % kBlk;
            const int x   = e % kBlk;
            const float* pr = P + (e / (kBlk * kBlk)) * kBlk * kBlk + r * kBlk;
            float val = pr[x];
            for (int mm = x + 1; mm < kBlk; ++mm) {
                val = fmaf(pr[mm], s.m[J * kBlk + mm][J * kBlk + x], val);
            }
            out[n] = val;
        }
        __syncthreads();
        for (int e = tid, n = 0; e < nb * kBlk * kBlk; e += kPrepThreads, ++n) {
            const int I = J + 1 + e / (kBlk * kBlk);
            const int r = (e / kBlk) % kBlk;
            const int x = e % kBlk;
            s.m[I * kBlk + r][J * kBlk + x] = out[n];
        }
        __syncthreads();
    }
    { // FP32 T -> FP16 T over the same storage: read everything, barrier, write.
        constexpr int kPer = kBT * kBT / kPrepThreads;
        __half tv[kPer];
#pragma unroll
        for (int n = 0; n < kPer; ++n) {
            const int idx = tid + n * kPrepThreads;
            tv[n]         = __float2half_rn(t_entry(s, idx / kBT, idx % kBT));
        }
        __syncthreads();
#pragma unroll
        for (int n = 0; n < kPer; ++n) {
            const int idx = tid + n * kPrepThreads;
            s.t[idx / kBT][idx % kBT] = tv[n];
        }
    }
    // b exp(G) K in place.
    for (int idx = tid; idx < kBT * kD; idx += kPrepThreads) {
        const int i = idx / kD;
        const int d = idx % kD;
        s.k[i][d] = __float2half_rn(__half2float(s.k[i][d]) * s.scale[i]);
    }
    __syncthreads();

    t_times_rows(s, W + (t0 * H_v + h) * kD, row_v);
    __syncthreads();
    load_rows_bf16(v + (t0 * H_v + h) * kD, row_v, s.k, s.b);
    __syncthreads();
    t_times_rows(s, U + (t0 * H_v + h) * kD, row_v);
}

// --------------------------------------------------------------------------------------------
// Stage 2: sequential state passing.  One CTA = one value head x a 32-column dv strip; the W,
// q and K^T operands come straight from the FP16 workspace. It also produces the inter-chunk
// output term exp(G_t) q_t h_c, so h_c never leaves the SM (the sm_80 route stores h_chunk and
// the output stage reads it back: 2x 32 KiB per chunk and head).
// --------------------------------------------------------------------------------------------
constexpr int kStrip        = 32;
constexpr int kStripLd      = kStrip + 4; // FP32
constexpr int kStripHd      = kStrip + 8; // FP16
constexpr int kStateThreads = 256;        // 8 warps: one output tile per warp in each product

struct StateSmem {
    float hf[kD][kStripLd]; // h[dk][dv] FP32, the running state
    union {
        __half hh[kD][kStripHd];  // FP16 copy of h_c, the B operand of W h_c and q h_c
        float tmp[kBT][kStripLd]; // then W h_c (hh is dead by the time it is stored)
    };
    float qh[kBT][kStripLd];  // q h_c (the inter-chunk output term, before exp(G_t))
    __half vd[kBT][kStripHd]; // exp(G_63 - G_i) v_new
    float g[kBT];
};

// 128 CTAs for the 35B-A3B shape (32 value heads x 4 strips), 2 per SM: one wave on a V100.
__global__ __launch_bounds__(kStateThreads, 2) void gdn_volta_state_kernel(
    const __half* __restrict__ W, const __half* __restrict__ U, const __half* __restrict__ k16,
    const __half* __restrict__ q16, const float* __restrict__ g_cumsum, const float* state_in,
    __half* __restrict__ v_new, __half* __restrict__ inter, float* state_out, int H_qk, int H_v,
    int chunks) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    StateSmem& s = *reinterpret_cast<StateSmem*>(smem_raw);
    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int strips = kD / kStrip;
    const int h      = static_cast<int>(blockIdx.x) / strips;
    const int dv0    = (static_cast<int>(blockIdx.x) % strips) * kStrip;
    const int qkh    = h / (H_v / H_qk);
    const std::int64_t state_base = static_cast<std::int64_t>(h) * kD * kD;
    const std::int64_t row_v      = static_cast<std::int64_t>(H_v) * kD;
    const std::int64_t row_qk     = static_cast<std::int64_t>(H_qk) * kD;

    // Read exactly this CTA's state elements (state_in may alias state_out).
    for (int idx = tid; idx < kD * kStrip; idx += kStateThreads) {
        const int dvl = idx / kD;
        const int dk  = idx % kD;
        s.hf[dk][dvl] = state_in[state_base + static_cast<std::int64_t>(dv0 + dvl) * kD + dk];
    }
    __syncthreads();

    for (int c = 0; c < chunks; ++c) {
        const std::int64_t t0 = static_cast<std::int64_t>(c) * kBT;
        if (tid < kBT) { s.g[tid] = g_cumsum[(t0 + tid) * H_v + h]; }
        for (int idx = tid; idx < kD * kStrip; idx += kStateThreads) {
            s.hh[idx / kStrip][idx % kStrip] = __float2half_rn(s.hf[idx / kStrip][idx % kStrip]);
        }
        __syncthreads();

        { // W h_c and q h_c, [64][32] each: warp -> (row tile warp&3, column tile warp>>2)
            const int ti = warp & 3;
            const int tj = warp >> 2;
            const __half* wrow = W + ((t0 + ti * 16) * H_v + h) * kD;
            const __half* qrow = q16 + (t0 + ti * 16) * row_qk + qkh * kD;
            FragAcc acc_w;
            FragAcc acc_q;
            wmma::fill_fragment(acc_w, 0.0f);
            wmma::fill_fragment(acc_q, 0.0f);
#pragma unroll
            for (int kk = 0; kk < kD / 16; ++kk) {
                FragA aw;
                FragA aq;
                FragB b;
                wmma::load_matrix_sync(b, &s.hh[kk * 16][tj * 16], kStripHd);
                wmma::load_matrix_sync(aw, wrow + kk * 16, static_cast<unsigned>(row_v));
                wmma::load_matrix_sync(aq, qrow + kk * 16, static_cast<unsigned>(row_qk));
                wmma::mma_sync(acc_w, aw, b, acc_w);
                wmma::mma_sync(acc_q, aq, b, acc_q);
            }
            __syncthreads(); // every warp is done reading hh, which tmp overlays
            wmma::store_matrix_sync(&s.tmp[ti * 16][tj * 16], acc_w, kStripLd,
                                    wmma::mem_row_major);
            wmma::store_matrix_sync(&s.qh[ti * 16][tj * 16], acc_q, kStripLd,
                                    wmma::mem_row_major);
        }
        __syncthreads();

        const float g_last = s.g[kBT - 1];
        for (int idx = tid; idx < kBT * kStrip; idx += kStateThreads) {
            const int i   = idx / kStrip;
            const int dvl = idx % kStrip;
            const std::int64_t at = (t0 + i) * row_v + h * kD + dv0 + dvl;
            const float vn = __half2float(U[at]) - s.tmp[i][dvl];
            v_new[at]      = __float2half_rn(vn);
            inter[at]      = __float2half_rn(s.qh[i][dvl] * expf(s.g[i]));
            s.vd[i][dvl]   = __float2half_rn(vn * expf(g_last - s.g[i]));
        }
        const float gamma = expf(g_last);
        for (int idx = tid; idx < kD * kStrip; idx += kStateThreads) {
            s.hf[idx / kStrip][idx % kStrip] *= gamma;
        }
        __syncthreads();

        { // h[128][32] += K^T[128][64] vd[64][32]: warp -> row tile `warp`, both column tiles;
          // K^T is read column-major straight from the FP16 copy of k.
            const int ti = warp;
            FragAcc acc[kStrip / 16];
#pragma unroll
            for (int tj = 0; tj < kStrip / 16; ++tj) {
                wmma::load_matrix_sync(acc[tj], &s.hf[ti * 16][tj * 16], kStripLd,
                                       wmma::mem_row_major);
            }
#pragma unroll
            for (int kk = 0; kk < kBT / 16; ++kk) {
                FragAT a;
                wmma::load_matrix_sync(a, k16 + (t0 + kk * 16) * row_qk + qkh * kD + ti * 16,
                                       static_cast<unsigned>(row_qk));
#pragma unroll
                for (int tj = 0; tj < kStrip / 16; ++tj) {
                    FragB b;
                    wmma::load_matrix_sync(b, &s.vd[kk * 16][tj * 16], kStripHd);
                    wmma::mma_sync(acc[tj], a, b, acc[tj]);
                }
            }
#pragma unroll
            for (int tj = 0; tj < kStrip / 16; ++tj) {
                wmma::store_matrix_sync(&s.hf[ti * 16][tj * 16], acc[tj], kStripLd,
                                        wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    for (int idx = tid; idx < kD * kStrip; idx += kStateThreads) {
        const int dvl = idx / kD;
        const int dk  = idx % kD;
        state_out[state_base + static_cast<std::int64_t>(dv0 + dvl) * kD + dk] = s.hf[dk][dvl];
    }
}

// --------------------------------------------------------------------------------------------
// Stage 3: output.  One CTA = one chunk x value head, 8 warps over the full 128 dv:
//   o = scale (inter + A v_new),  A[t][s] = (q_t.k_s) exp(G_t - G_s) for s <= t (diagonal in).
// q and k are staged with coalesced loads (their buffers are then reused for v_new and the
// per-warp epilogue staging); A is built tile by tile from the q k^T accumulators.
// --------------------------------------------------------------------------------------------
constexpr int kOutThreads = 256;

struct OutputSmem {
    __half qv[kBT][kHalfLd];   // q, then v_new
    union {
        __half k[kBT][kHalfLd]; // k
        float stage[8][16 * 16]; // then per-warp FP32 staging
    };
    __half a[kBT][kTileLd];    // masked, gated q k^T
    float g[kBT];
};

__device__ __forceinline__ void load_rows_f16(const __half* __restrict__ src,
                                              std::int64_t row_stride, __half (*dst)[kHalfLd]) {
    for (int idx = static_cast<int>(threadIdx.x); idx < kBT * (kD / 8);
         idx += static_cast<int>(blockDim.x)) {
        const int row = idx / (kD / 8);
        const int col = (idx % (kD / 8)) * 8;
        *reinterpret_cast<uint4*>(&dst[row][col]) =
            *reinterpret_cast<const uint4*>(src + row * row_stride + col);
    }
}

__global__ __launch_bounds__(kOutThreads, 2) void gdn_volta_output_kernel(
    const __half* __restrict__ q16, const __half* __restrict__ k16,
    const __half* __restrict__ v_new, const __half* __restrict__ inter,
    const float* __restrict__ g_cumsum, __nv_bfloat16* __restrict__ out, int H_qk, int H_v,
    float scale) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    OutputSmem& s = *reinterpret_cast<OutputSmem*>(smem_raw);
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int c    = static_cast<int>(blockIdx.x);
    const int h    = static_cast<int>(blockIdx.y);
    const int qh   = h / (H_v / H_qk);
    const std::int64_t t0     = static_cast<std::int64_t>(c) * kBT;
    const std::int64_t row_qk = static_cast<std::int64_t>(H_qk) * kD;
    const std::int64_t row_v  = static_cast<std::int64_t>(H_v) * kD;

    if (tid < kBT) { s.g[tid] = g_cumsum[(t0 + tid) * H_v + h]; }
    load_rows_f16(q16 + t0 * row_qk + qh * kD, row_qk, s.qv);
    load_rows_f16(k16 + t0 * row_qk + qh * kD, row_qk, s.k);
    __syncthreads();

    // q k^T on the lower block triangle: 10 tiles, at most 2 per warp, held in registers until
    // k is dead, then staged per warp (over k's storage) and turned into A.
    FragAcc sacc[2];
    int stile[2] = {-1, -1};
    for (int n = 0, tile = warp; tile < 10; ++n, tile += kOutThreads / 32) {
        int ti = 0;
        while ((ti + 1) * (ti + 2) / 2 <= tile) { ++ti; }
        const int tj = tile - ti * (ti + 1) / 2;
        wmma::fill_fragment(sacc[n], 0.0f);
#pragma unroll
        for (int kk = 0; kk < kD / 16; ++kk) {
            FragA a;
            FragBT b;
            wmma::load_matrix_sync(a, &s.qv[ti * 16][kk * 16], kHalfLd);
            wmma::load_matrix_sync(b, &s.k[tj * 16][kk * 16], kHalfLd);
            wmma::mma_sync(sacc[n], a, b, sacc[n]);
        }
        stile[n] = ti * 4 + tj;
    }
    __syncthreads();
#pragma unroll
    for (int n = 0; n < 2; ++n) {
        if (stile[n] < 0) { continue; }
        const int ti = stile[n] / 4;
        const int tj = stile[n] % 4;
        wmma::store_matrix_sync(s.stage[warp], sacc[n], 16, wmma::mem_row_major);
        __syncwarp();
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int e  = lane + 32 * r;
            const int t  = ti * 16 + e / 16;
            const int sj = tj * 16 + e % 16;
            // Diagonal included; a select keeps exp(G_t - G_s) for s > t (may be +inf) out.
            s.a[t][sj] = __float2half_rn(sj <= t ? s.stage[warp][e] * expf(s.g[t] - s.g[sj])
                                                 : 0.0f);
        }
        __syncwarp();
    }
    // Upper block triangle of A is zero.
    for (int idx = tid; idx < kBT * kBT; idx += kOutThreads) {
        const int t  = idx / kBT;
        const int sj = idx % kBT;
        if (sj / 16 > t / 16) { s.a[t][sj] = __float2half_rn(0.0f); }
    }
    __syncthreads();
    load_rows_f16(v_new + t0 * row_v + h * kD, row_v, s.qv);
    __syncthreads();

    // A v_new [64][128]: 32 tiles, 4 per warp (row tile warp&3, column tiles (warp>>2)*4 + j).
    const int ti = warp & 3;
    FragAcc acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) { wmma::fill_fragment(acc[j], 0.0f); }
    for (int kk = 0; kk <= ti; ++kk) { // A is lower triangular
        FragA a;
        wmma::load_matrix_sync(a, &s.a[ti * 16][kk * 16], kTileLd);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            FragB b;
            wmma::load_matrix_sync(b, &s.qv[kk * 16][((warp >> 2) * 4 + j) * 16], kHalfLd);
            wmma::mma_sync(acc[j], a, b, acc[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int tj = (warp >> 2) * 4 + j;
        wmma::store_matrix_sync(s.stage[warp], acc[j], 16, wmma::mem_row_major);
        __syncwarp();
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            const int e  = lane + 32 * n;
            const std::int64_t at = (t0 + ti * 16 + e / 16) * row_v + h * kD + tj * 16 + e % 16;
            out[at] = __float2bfloat16_rn(scale * (s.stage[warp][e] + __half2float(inter[at])));
        }
        __syncwarp();
    }
}

template <class Kernel>
void set_smem(Kernel kernel, std::size_t bytes) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(bytes)));
    // Occupancy here is bounded by shared memory; take the whole 96 KiB carveout.
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                    cudaSharedmemCarveoutMaxShared));
}

} // namespace

void launch_chunked(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                    const Tensor& beta, float scale, const Tensor& ssm_state_in,
                    Tensor& ssm_state_out, Tensor& out, void* workspace,
                    std::size_t workspace_bytes, cudaStream_t stream) {
    const std::int32_t H_qk = q.ne[1];
    const std::int32_t H_v  = v.ne[1];
    const std::int32_t L    = q.ne[2];
    if (L <= 0 || L % kBT != 0) {
        throw std::invalid_argument("gated_delta_net chunked (sm_70): L must be a multiple of 64");
    }
    const auto layout = chunked::compute_workspace_layout(H_v, L);
    if (workspace == nullptr || workspace_bytes < layout.total_bytes) { throw std::bad_alloc(); }
    const DeviceSpan backing{workspace, workspace_bytes};
    auto* g_cumsum = static_cast<float*>(layout.g_cumsum.bind(backing).data);
    // W, U, v_new and h_chunk carry FP16 in their BF16-typed regions (private to this route).
    auto* W       = static_cast<__half*>(layout.W.bind(backing).data);
    auto* U       = static_cast<__half*>(layout.U.bind(backing).data);
    auto* v_new   = static_cast<__half*>(layout.v_new.bind(backing).data);
    // The h_chunk region holds the inter-chunk output term exp(G_t) q_t h_c instead, FP16
    // [128, H_v, L] (half the region): the state stage computes it while h_c is on chip.
    auto* inter   = static_cast<__half*>(layout.h_chunk.bind(backing).data);
    auto* q16     = static_cast<__half*>(layout.q16.bind(backing).data);
    auto* k16     = static_cast<__half*>(layout.k16.bind(backing).data);
    const int chunks = L / kBT;

    static const bool configured = [] {
        set_smem(gdn_volta_prepare_kernel, sizeof(PrepareSmem));
        set_smem(gdn_volta_state_kernel, sizeof(StateSmem));
        set_smem(gdn_volta_output_kernel, sizeof(OutputSmem));
        return true;
    }();
    (void)configured;

    gdn_volta_prepare_kernel<<<dim3(chunks, H_v), kPrepThreads, sizeof(PrepareSmem), stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data), W, U, g_cumsum, q16, k16, H_qk, H_v);
    CUDA_CHECK(cudaGetLastError());
    gdn_volta_state_kernel<<<H_v * (kD / kStrip), kStateThreads, sizeof(StateSmem), stream>>>(
        W, U, k16, q16, g_cumsum, static_cast<const float*>(ssm_state_in.data), v_new, inter,
        static_cast<float*>(ssm_state_out.data), H_qk, H_v, chunks);
    CUDA_CHECK(cudaGetLastError());
    gdn_volta_output_kernel<<<dim3(chunks, H_v), kOutThreads, sizeof(OutputSmem), stream>>>(
        q16, k16, v_new, inter, g_cumsum, static_cast<__nv_bfloat16*>(out.data), H_qk, H_v,
        scale);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::gated_delta_net
