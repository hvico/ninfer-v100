#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/common/volta_mma.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/w8/w8_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"
#include "ops/sparse_moe/decode/sparse_moe_decode.h"
#include "ops/sparse_moe/sparse_moe_route.cuh"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden       = 2048;
constexpr int kExperts      = 256;
constexpr int kRouterRows   = 257;
constexpr int kTopK         = 8;
constexpr int kIntermediate = 512;

constexpr int kRouterBM      = 16;
constexpr int kRouterBN      = 64;
constexpr int kRouterBK      = 64;
constexpr int kRouterStages  = 2;
constexpr int kRouterWarps   = 8;
constexpr int kRouterThreads = 32 * kRouterWarps;

__global__ __launch_bounds__(kRouterThreads, 2) void sparse_moe_prefill_router_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ scores, int tokens) {
    __shared__ __align__(16) __nv_bfloat16 Ws[kRouterStages][kRouterBM * kRouterBK];
    __shared__ __align__(16) __nv_bfloat16 Xs[kRouterStages][kRouterBN * kRouterBK];

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int row0   = static_cast<int>(blockIdx.x) * kRouterBM;
    const int token0 = static_cast<int>(blockIdx.y) * kRouterBN;

    float acc[4] = {};

    auto stage_inputs = [&](int stage, int kt) {
        const int k0 = kt * kRouterBK;
        for (int item = tid; item < kRouterBM * (kRouterBK / 8); item += kRouterThreads) {
            const int row   = item / (kRouterBK / 8);
            const int k8    = item - row * (kRouterBK / 8);
            const int rr    = row0 + row;
            auto* dst       = &Ws[stage][row * kRouterBK + gemm_swz64(row, k8 * 8)];
            const auto* src = weight +
                              static_cast<std::int64_t>(rr < kRouterRows ? rr : 0) * kHidden + k0 +
                              k8 * 8;
            cp_async_zfill<16, Cache::cg>(dst, src, rr < kRouterRows ? 16 : 0);
        }
        for (int item = tid; item < kRouterBN * (kRouterBK / 8); item += kRouterThreads) {
            const int col = item / (kRouterBK / 8);
            const int k8  = item - col * (kRouterBK / 8);
            const int tt  = token0 + col;
            auto* dst     = &Xs[stage][col * kRouterBK + gemm_swz64(col, k8 * 8)];
            const auto* src =
                x + static_cast<std::int64_t>(tt < tokens ? tt : 0) * kHidden + k0 + k8 * 8;
            cp_async_zfill<16, Cache::cg>(dst, src, tt < tokens ? 16 : 0);
        }
    };

#pragma unroll
    for (int stage = 0; stage < kRouterStages; ++stage) {
        stage_inputs(stage, stage);
        cp_commit();
    }

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

#pragma unroll 1
    for (int kt = 0; kt < kHidden / kRouterBK; ++kt) {
        const int stage = kt & 1;
        cp_wait<kRouterStages - 1>();
        __syncthreads();

#pragma unroll
        for (int ki = 0; ki < kRouterBK / 16; ++ki) {
            unsigned af[4];
            unsigned bf[2];
            const int arow = a_rowoff;
            const int acol = ki * 16 + a_coloff;
            const int brow = warp * 8 + b_rin;
            const int bcol = ki * 16 + b_koff;
            ldmatrix_x4(af[0], af[1], af[2], af[3],
                        smem_addr(&Ws[stage][arow * kRouterBK + gemm_swz64(arow, acol)]));
            ldmatrix_x2(bf[0], bf[1],
                        smem_addr(&Xs[stage][brow * kRouterBK + gemm_swz64(brow, bcol)]));
            mma_bf16(acc[0], acc[1], acc[2], acc[3], af[0], af[1], af[2], af[3], bf[0], bf[1]);
        }

        __syncthreads();
        const int next = kt + kRouterStages;
        if (next < kHidden / kRouterBK) { stage_inputs(stage, next); }
        cp_commit();
    }
    cp_wait<0>();

    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int r0  = row0 + gid;
    const int r1  = r0 + 8;
    const int c0  = token0 + warp * 8 + 2 * lid;
    const int c1  = c0 + 1;
    if (r0 < kRouterRows && c0 < tokens) {
        scores[static_cast<std::int64_t>(c0) * kSparseMoeRouterScoreRows + r0] = acc[0];
    }
    if (r0 < kRouterRows && c1 < tokens) {
        scores[static_cast<std::int64_t>(c1) * kSparseMoeRouterScoreRows + r0] = acc[1];
    }
    if (r1 < kRouterRows && c0 < tokens) {
        scores[static_cast<std::int64_t>(c0) * kSparseMoeRouterScoreRows + r1] = acc[2];
    }
    if (r1 < kRouterRows && c1 < tokens) {
        scores[static_cast<std::int64_t>(c1) * kSparseMoeRouterScoreRows + r1] = acc[3];
    }
}

__global__ void sparse_moe_prefill_select_count_kernel(const float* __restrict__ scores,
                                                       int* __restrict__ ids,
                                                       float* __restrict__ alpha,
                                                       float* __restrict__ shared_scale,
                                                       int* __restrict__ local_rank,
                                                       int* __restrict__ tile_counts, int tokens) {
    __shared__ int counts[kExperts];
    __shared__ float selected_logits[kSparseMoeRouteTileTokens][kTopK];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid < kExperts) { counts[tid] = 0; }
    __syncthreads();

    const int token = static_cast<int>(blockIdx.x) * kSparseMoeRouteTileTokens + warp;
    if (token < tokens) {
        sparse_moe_select_top8_warp(scores + static_cast<std::int64_t>(token) *
                                                 kSparseMoeRouterScoreRows,
                                    ids + token * kTopK, alpha + token * kTopK,
                                    shared_scale + token, selected_logits[warp]);
        __syncwarp();
        if (lane == 0) {
#pragma unroll
            for (int rank = 0; rank < kTopK; ++rank) {
                const int assignment   = token * kTopK + rank;
                local_rank[assignment] = atomicAdd(&counts[ids[assignment]], 1);
            }
        }
    }
    __syncthreads();
    if (tid < kExperts) {
        tile_counts[static_cast<std::int64_t>(blockIdx.x) * kExperts + tid] = counts[tid];
    }
}

__global__ void sparse_moe_prefill_scan_kernel(const int* __restrict__ tile_counts,
                                               int* __restrict__ tile_bases,
                                               int* __restrict__ expert_offsets,
                                               int* __restrict__ route_job_experts,
                                               int* __restrict__ route_job_columns,
                                               int* __restrict__ route_job_count, int route_tiles,
                                               int job_bn, int tokens, bool adaptive) {
    __shared__ int scan[kExperts];
    const int expert = static_cast<int>(threadIdx.x);
    int count        = 0;
    for (int tile = 0; tile < route_tiles; ++tile) {
        count += tile_counts[static_cast<std::int64_t>(tile) * kExperts + expert];
    }
    scan[expert] = count;
    __syncthreads();

#pragma unroll
    for (int offset = 1; offset < kExperts; offset <<= 1) {
        const int add = expert >= offset ? scan[expert - offset] : 0;
        __syncthreads();
        scan[expert] += add;
        __syncthreads();
    }

    const int base         = expert == 0 ? 0 : scan[expert - 1];
    expert_offsets[expert] = base;
    if (expert == kExperts - 1) { expert_offsets[kExperts] = scan[expert]; }

    int cursor = base;
    for (int tile = 0; tile < route_tiles; ++tile) {
        const std::int64_t index = static_cast<std::int64_t>(tile) * kExperts + expert;
        tile_bases[index]        = cursor;
        cursor += tile_counts[index];
    }

    // Reuse scan only after every expert has consumed the assignment prefix.
    __syncthreads();
    scan[expert] = (count + job_bn - 1) / job_bn;
    __syncthreads();
#pragma unroll
    for (int offset = 1; offset < kExperts; offset <<= 1) {
        const int add = expert >= offset ? scan[expert - offset] : 0;
        __syncthreads();
        scan[expert] += add;
        __syncthreads();
    }
    const int job_begin = expert == 0 ? 0 : scan[expert - 1];
    const int jobs      = (count + job_bn - 1) / job_bn;
    for (int job = 0; job < jobs; ++job) {
        route_job_experts[job_begin + job] = expert;
        route_job_columns[job_begin + job] = job * job_bn;
    }
    if (expert == kExperts - 1) {
        const int jobs = scan[expert];
        // Above 3.5 grouped jobs per token, fixed token work wins by avoiding sparse expert tiles.
        *route_job_count = adaptive && jobs * 2 > 7 * tokens ? -jobs : jobs;
    }
}

template <bool Adaptive>
__global__ void
sparse_moe_prefill_gather_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids,
                                 const int* __restrict__ local_rank,
                                 int* __restrict__ packed_index, const int* __restrict__ tile_bases,
                                 __nv_bfloat16* __restrict__ gathered,
                                 const int* __restrict__ route_job_count) {
    if constexpr (Adaptive) {
        if (*route_job_count < 0) { return; }
    }
    const int assignment = static_cast<int>(blockIdx.x);
    const int token      = assignment / kTopK;
    const int expert     = ids[assignment];
    const int tile       = token / kSparseMoeRouteTileTokens;
    const int packed =
        tile_bases[static_cast<std::int64_t>(tile) * kExperts + expert] + local_rank[assignment];
    const int k       = static_cast<int>(threadIdx.x) * 8;
    const uint4 value = load_vec<uint4>(x + static_cast<std::int64_t>(token) * kHidden + k);
    store_vec(gathered + static_cast<std::int64_t>(packed) * kHidden + k, value);
    if (threadIdx.x == 0) { packed_index[assignment] = packed; }
}

// Publishes the packed-column map without moving activations. One thread per assignment, so
// unlike the gather it reads the tile-local rank and writes the inverse map from the same
// thread and never has the two live in one buffer.
template <bool Adaptive>
__global__ void
sparse_moe_prefill_index_kernel(const int* __restrict__ ids, const int* __restrict__ local_rank,
                                int* __restrict__ packed_index, const int* __restrict__ tile_bases,
                                int* __restrict__ packed_token, int assignments,
                                const int* __restrict__ route_job_count) {
    if constexpr (Adaptive) {
        if (*route_job_count < 0) { return; }
    }
    const int assignment =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (assignment >= assignments) { return; }
    const int token  = assignment / kTopK;
    const int expert = ids[assignment];
    const int tile   = token / kSparseMoeRouteTileTokens;
    const int packed =
        tile_bases[static_cast<std::int64_t>(tile) * kExperts + expert] + local_rank[assignment];
    packed_index[assignment] = packed;
    packed_token[packed]     = token;
}

constexpr int kExpertBM                = 64;
constexpr int kExpertBN                = 64;
constexpr int kExpertBK                = 64;
constexpr int kExpertStages            = 2;
constexpr int kExpertWarps             = 8;
constexpr int kExpertThreads           = 32 * kExpertWarps;
constexpr int kRtx5090SmCount          = 170;
constexpr int kPrefillBlocksPerSm      = 3;
constexpr int kPrefillPersistentBlocks = kPrefillBlocksPerSm * kRtx5090SmCount;

template <int ExpertWarps, int ExpertBN>
__global__ __launch_bounds__(ExpertWarps * 32, 3) void sparse_moe_prefill_q4_gate_up_kernel(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ packed_token,
    const int* __restrict__ expert_offsets, const int* __restrict__ route_job_experts,
    const int* __restrict__ route_job_columns, const int* __restrict__ route_job_count,
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ activation) {
    constexpr int ExpertThreads = ExpertWarps * 32;
    constexpr int GroupsPerRow  = kHidden / 64;
    constexpr int WarpCols      = ExpertBN / ExpertWarps;
    constexpr int WarpNT        = WarpCols / 8;
    constexpr int StageChunks   = kExpertBK / 8;
    constexpr int StageIters    = ExpertBN * StageChunks / ExpertThreads;
    static_assert(ExpertBN % ExpertWarps == 0 && WarpCols % 8 == 0);
    static_assert(StageIters * ExpertThreads == ExpertBN * StageChunks,
                  "the staging loop is unrolled, so every thread must take the same column count");
    __shared__ __align__(16) __nv_bfloat16 As[kExpertBM * kExpertBK];
    __shared__ __align__(16) __nv_bfloat16 Bs[kExpertStages][ExpertBN * kExpertBK];
    __shared__ __align__(16) std::uint8_t Cr[kExpertStages][kExpertBM * 32];
    __shared__ __align__(16) std::uint8_t Sr[kExpertBM * GroupsPerRow * 2];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int a_mat          = lane >> 3;
    const int a_rin          = lane & 7;
    const int a_rowoff       = a_rin + ((a_mat & 1) << 3);
    const int a_coloff       = (a_mat >> 1) << 3;
    const int b_rin          = lane & 7;
    const int b_koff         = ((lane >> 3) & 1) << 3;
    const int gid            = lane >> 2;
    const int lid            = lane & 3;
    constexpr int row_blocks = kIntermediate / (kExpertBM / 2);
    const int total_work     = *route_job_count * row_blocks;
    for (int work = static_cast<int>(blockIdx.x); work < total_work;
         work += static_cast<int>(gridDim.x)) {
        const int route_job   = work / row_blocks;
        const int row_block   = work - route_job * row_blocks;
        const int expert      = route_job_experts[route_job];
        const int logical0    = row_block * (kExpertBM / 2);
        const int begin       = expert_offsets[expert];
        const int count       = expert_offsets[expert + 1] - begin;
        const int column_base = route_job_columns[route_job];
        const int cols        = count - column_base < ExpertBN ? count - column_base : ExpertBN;
        // A thread stages the same columns of all kHidden / kExpertBK tiles, so it resolves their
        // rows once for the whole job instead of reading the map and redoing the row multiply per
        // tile. A column past the tile is clamped onto a live one and then zero-filled.
        // ptxas allots exactly 80 registers per thread at __launch_bounds__(256, 3) on
        // sm_120a and this kernel now uses all of them, so widening this array spills.
        const __nv_bfloat16* src_row[StageIters];
#pragma unroll
        for (int i = 0; i < StageIters; ++i) {
            const int col = (tid + i * ExpertThreads) / StageChunks;
            const int row = packed_token[col < cols ? begin + column_base + col : begin];
            src_row[i]    = x + static_cast<std::int64_t>(row) * kHidden;
        }
        float acc[4][WarpNT][4] = {};

        auto global_row = [&](int local_row) {
            const int logical = logical0 + (local_row & (kExpertBM / 2 - 1));
            return expert * 1024 + logical + (local_row >= kExpertBM / 2 ? kIntermediate : 0);
        };

        auto stage_scales = [&] {
            constexpr int ScaleBytesPerRow = GroupsPerRow * 2;
            constexpr int ChunksPerRow     = ScaleBytesPerRow / 16;
            for (int item = tid; item < kExpertBM * ChunksPerRow; item += ExpertThreads) {
                const int row   = item / ChunksPerRow;
                const int chunk = item - row * ChunksPerRow;
                const std::int64_t gi =
                    static_cast<std::int64_t>(global_row(row)) * GroupsPerRow + chunk * 8;
                cp_async<16, Cache::cg>(&Sr[row * ScaleBytesPerRow + chunk * 16], &scales[gi * 2]);
            }
        };

        auto stage_inputs = [&](int stage, int kt) {
            const int k0 = kt * kExpertBK;
#pragma unroll
            for (int i = 0; i < StageIters; ++i) {
                const int item = tid + i * ExpertThreads;
                const int col  = item / StageChunks;
                const int k8   = item - col * StageChunks;
                auto* dst      = &Bs[stage][col * kExpertBK + gemm_swz64(col, k8 * 8)];
                cp_async_zfill<16, Cache::cg>(dst, src_row[i] + k0 + k8 * 8, col < cols ? 16 : 0);
            }

            const int group = kt;
            for (int item = tid; item < kExpertBM * 2; item += ExpertThreads) {
                const int row  = item >> 1;
                const int half = item & 1;
                const std::int64_t gi =
                    static_cast<std::int64_t>(global_row(row)) * GroupsPerRow + group;
                cp_async<16, Cache::cg>(&Cr[stage][row * 32 + half * 16],
                                        &codes[gi * 32 + half * 16]);
            }
        };

        auto decode_weight = [&](int stage) {
            constexpr int CodeChunksPerRow = Q4RowSplitStorage::kCodeBytesPerGroup / 4;
            static_assert(CodeChunksPerRow * 8 == kExpertBK,
                          "a row of codes must decode to exactly the tile's k width");
            for (int item = tid; item < kExpertBM * CodeChunksPerRow; item += ExpertThreads) {
                const int row   = item / CodeChunksPerRow;
                const int chunk = item - row * CodeChunksPerRow;
                unsigned decoded[4];
                Q4MmaDecodeAtom::decode_eight(
                    *reinterpret_cast<const unsigned*>(&Cr[stage][row * 32 + chunk * 4]), decoded);
                store_vec(&As[row * kExpertBK + gemm_swz64(row, chunk * 8)],
                          make_int4(static_cast<int>(decoded[0]), static_cast<int>(decoded[1]),
                                    static_cast<int>(decoded[2]), static_cast<int>(decoded[3])));
            }
        };

        stage_scales();
        cp_commit();
#pragma unroll
        for (int stage = 0; stage < kExpertStages; ++stage) {
            stage_inputs(stage, stage);
            cp_commit();
        }

#pragma unroll 1
        for (int kt = 0; kt < kHidden / kExpertBK; ++kt) {
            const int stage = kt & 1;
            cp_wait<kExpertStages - 1>();
            __syncthreads();
            decode_weight(stage);
            __syncthreads();

            if (warp * WarpCols < cols) {
                float partial[4][WarpNT][4] = {};
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    unsigned af[4][4];
                    unsigned bf[WarpNT][2];
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                    }
#pragma unroll
                    for (int ni = 0; ni < WarpNT; ++ni) {
                        const int brow = warp * WarpCols + ni * 8 + b_rin;
                        const int bcol = ki * 16 + b_koff;
                        ldmatrix_x2(
                            bf[ni][0], bf[ni][1],
                            smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
#pragma unroll
                        for (int mi = 0; mi < 4; ++mi) {
                            mma_bf16(partial[mi][ni][0], partial[mi][ni][1], partial[mi][ni][2],
                                     partial[mi][ni][3], af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                     bf[ni][0], bf[ni][1]);
                        }
                    }
                }
#pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    const int row0 = mi * 16 + gid;
                    const int row1 = row0 + 8;
                    float scale0 =
                        lid == 0
                            ? __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                                  &Sr[(row0 * GroupsPerRow + kt) * 2])))
                            : 0.0f;
                    float scale1 =
                        lid == 0
                            ? __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                                  &Sr[(row1 * GroupsPerRow + kt) * 2])))
                            : 0.0f;
                    scale0 = __shfl_sync(0xffffffffu, scale0, gid * 4);
                    scale1 = __shfl_sync(0xffffffffu, scale1, gid * 4);
#pragma unroll
                    for (int ni = 0; ni < WarpNT; ++ni) {
                        acc[mi][ni][0] = fmaf(partial[mi][ni][0], scale0, acc[mi][ni][0]);
                        acc[mi][ni][1] = fmaf(partial[mi][ni][1], scale0, acc[mi][ni][1]);
                        acc[mi][ni][2] = fmaf(partial[mi][ni][2], scale1, acc[mi][ni][2]);
                        acc[mi][ni][3] = fmaf(partial[mi][ni][3], scale1, acc[mi][ni][3]);
                    }
                }
            }

            __syncthreads();
            const int next = kt + kExpertStages;
            if (next < kHidden / kExpertBK) { stage_inputs(stage, next); }
            cp_commit();
        }
        cp_wait<0>();
        __syncthreads();

        if (warp * WarpCols < cols) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int row0 = logical0 + mi * 16 + gid;
                const int row1 = row0 + 8;
#pragma unroll
                for (int ni = 0; ni < WarpNT; ++ni) {
                    const int col0       = begin + column_base + warp * WarpCols + ni * 8 + 2 * lid;
                    const int col1       = col0 + 1;
                    const int local_col0 = warp * WarpCols + ni * 8 + 2 * lid;
                    const int local_col1 = local_col0 + 1;
                    if (local_col0 < cols) {
                        activation[static_cast<std::int64_t>(col0) * kIntermediate + row0] =
                            __float2bfloat16_rn(silu(acc[mi][ni][0]) * acc[mi + 2][ni][0]);
                        activation[static_cast<std::int64_t>(col0) * kIntermediate + row1] =
                            __float2bfloat16_rn(silu(acc[mi][ni][2]) * acc[mi + 2][ni][2]);
                    }
                    if (local_col1 < cols) {
                        activation[static_cast<std::int64_t>(col1) * kIntermediate + row0] =
                            __float2bfloat16_rn(silu(acc[mi][ni][1]) * acc[mi + 2][ni][1]);
                        activation[static_cast<std::int64_t>(col1) * kIntermediate + row1] =
                            __float2bfloat16_rn(silu(acc[mi][ni][3]) * acc[mi + 2][ni][3]);
                    }
                }
            }
        }
        __syncthreads();
    }
}

template <bool Routed, bool Adaptive = false>
__global__ __launch_bounds__(kExpertThreads, 1) void sparse_moe_prefill_w8_gate_up_kernel(
    const __nv_bfloat16* __restrict__ input, const int* __restrict__ expert_offsets,
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ activation, int tokens, const int* __restrict__ route_job_count) {
    __shared__ __align__(16) __nv_bfloat16 As[kExpertBM * kExpertBK];
    __shared__ __align__(16) __nv_bfloat16 Bs[kExpertStages][kExpertBN * kExpertBK];
    __shared__ __align__(16) std::uint8_t Cr[kExpertBM * kExpertBK];
    __shared__ __align__(16) std::uint8_t Sr[kExpertBM * 16];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if constexpr (Adaptive) {
        if (*route_job_count < 0) { return; }
    }
    const int expert   = Routed ? static_cast<int>(blockIdx.y) : 0;
    const int logical0 = static_cast<int>(blockIdx.x) * (kExpertBM / 2);
    const int begin    = Routed ? expert_offsets[expert] : static_cast<int>(blockIdx.y) * kExpertBN;
    const int count    = Routed ? expert_offsets[expert + 1] - begin
                                : (tokens - begin < kExpertBN ? tokens - begin : kExpertBN);
    if (count <= 0) { return; }

    const int a_mat              = lane >> 3;
    const int a_rin              = lane & 7;
    const int a_rowoff           = a_rin + ((a_mat & 1) << 3);
    const int a_coloff           = (a_mat >> 1) << 3;
    const int b_rin              = lane & 7;
    const int b_koff             = ((lane >> 3) & 1) << 3;
    const int gid                = lane >> 2;
    const int lid                = lane & 3;
    constexpr int groups_per_row = kHidden / 32;

    const int column_iterations = Routed ? (count + kExpertBN - 1) / kExpertBN : 1;
    for (int iteration = 0; iteration < column_iterations; ++iteration) {
        const int column_base = iteration * kExpertBN;
        const int cols        = count - column_base < kExpertBN ? count - column_base : kExpertBN;
        float acc[4][4]       = {};

        auto global_row = [&](int local_row) {
            const int logical = logical0 + (local_row & (kExpertBM / 2 - 1));
            const int row     = logical + (local_row >= kExpertBM / 2 ? kIntermediate : 0);
            return (Routed ? expert * 1024 : 0) + row;
        };

        auto stage_x = [&](int stage, int kt) {
            const int k0 = kt * kExpertBK;
            for (int item = tid; item < kExpertBN * (kExpertBK / 8); item += kExpertThreads) {
                const int col        = item / (kExpertBK / 8);
                const int k8         = item - col * (kExpertBK / 8);
                const int source_col = begin + column_base + col;
                auto* dst            = &Bs[stage][col * kExpertBK + gemm_swz64(col, k8 * 8)];
                const auto* src =
                    input + static_cast<std::int64_t>(col < cols ? source_col : begin) * kHidden +
                    k0 + k8 * 8;
                cp_async_zfill<16, Cache::cg>(dst, src, col < cols ? 16 : 0);
            }
        };

        auto stage_weight = [&](int kt) {
            constexpr int groups_per_tile   = kExpertBK / 32;
            constexpr int scale_cache_tiles = 8 / groups_per_tile;
            const int group0                = kt * groups_per_tile;
            for (int item = tid; item < kExpertBM * (kExpertBK / 16); item += kExpertThreads) {
                const int row   = item / (kExpertBK / 16);
                const int chunk = item - row * (kExpertBK / 16);
                const std::int64_t gi =
                    static_cast<std::int64_t>(global_row(row)) * groups_per_row + group0;
                cp_async<16, Cache::cg>(&Cr[row * kExpertBK + chunk * 16],
                                        &codes[gi * 32 + chunk * 16]);
            }
            if ((kt % scale_cache_tiles) == 0) {
                for (int row = tid; row < kExpertBM; row += kExpertThreads) {
                    const std::int64_t gi =
                        static_cast<std::int64_t>(global_row(row)) * groups_per_row + group0;
                    cp_async<16, Cache::cg>(&Sr[row * 16], &scales[gi * 2]);
                }
            }
        };

        auto decode_weight = [&](int kt) {
            constexpr int groups_per_tile   = kExpertBK / 32;
            constexpr int scale_cache_tiles = 8 / groups_per_tile;
            const int scale_offset          = (kt % scale_cache_tiles) * groups_per_tile * 2;
            const int half                  = lane >> 4;
            const int half_lane             = lane & 15;
            for (int row_pair = warp * 2; row_pair < kExpertBM; row_pair += kExpertWarps * 2) {
                const int row = row_pair + half;
                unsigned scale_pair =
                    half_lane == 0
                        ? *reinterpret_cast<const std::uint32_t*>(&Sr[row * 16 + scale_offset])
                        : 0;
                scale_pair = __shfl_sync(0xffffffffu, scale_pair, half * 16);
#pragma unroll
                for (int group = 0; group < groups_per_tile; ++group) {
                    const float scale = __half2float(__ushort_as_half(
                        static_cast<std::uint16_t>((scale_pair >> (group * 16)) & 0xffffu)));
                    const int col     = group * 32 + half_lane * 2;
                    const std::uint16_t packed =
                        *reinterpret_cast<const std::uint16_t*>(&Cr[row * kExpertBK + col]);
                    const int q0 = static_cast<int>(static_cast<std::int8_t>(packed & 0xffu));
                    const int q1 = static_cast<int>(static_cast<std::int8_t>(packed >> 8));
                    const __nv_bfloat162 value = __floats2bfloat162_rn(
                        static_cast<float>(q0) * scale, static_cast<float>(q1) * scale);
                    store_vec(&As[row * kExpertBK + gemm_swz64(row, col)], value);
                }
            }
        };

        stage_x(0, 0);
        stage_weight(0);
        cp_commit();

#pragma unroll 4
        for (int kt = 0; kt < kHidden / kExpertBK; ++kt) {
            const int stage = kt & 1;
            cp_wait<0>();
            __syncthreads();
            decode_weight(kt);
            __syncthreads();

            const int next = kt + 1;
            if (next < kHidden / kExpertBK) {
                stage_x(next & 1, next);
                stage_weight(next);
                cp_commit();
            }

            if (warp * 8 < cols) {
                unsigned af[2][4][4];
                unsigned bf[2][2];
                auto load_fragments = [&](int slot, int ki) {
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[slot][mi][0], af[slot][mi][1], af[slot][mi][2],
                                    af[slot][mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                    }
                    const int brow = warp * 8 + b_rin;
                    const int bcol = ki * 16 + b_koff;
                    ldmatrix_x2(bf[slot][0], bf[slot][1],
                                smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
                };
                load_fragments(0, 0);
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    const int slot = ki & 1;
                    if (ki + 1 < kExpertBK / 16) { load_fragments(slot ^ 1, ki + 1); }
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        mma_bf16(acc[mi][0], acc[mi][1], acc[mi][2], acc[mi][3], af[slot][mi][0],
                                 af[slot][mi][1], af[slot][mi][2], af[slot][mi][3], bf[slot][0],
                                 bf[slot][1]);
                    }
                }
            }
        }
        cp_wait<0>();
        __syncthreads();

        if (warp * 8 < cols) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int row0       = logical0 + mi * 16 + gid;
                const int row1       = row0 + 8;
                const int col0       = begin + column_base + warp * 8 + 2 * lid;
                const int col1       = col0 + 1;
                const int local_col0 = warp * 8 + 2 * lid;
                const int local_col1 = local_col0 + 1;
                if (local_col0 < cols) {
                    activation[static_cast<std::int64_t>(col0) * kIntermediate + row0] =
                        __float2bfloat16_rn(silu(acc[mi][0]) * acc[mi + 2][0]);
                    activation[static_cast<std::int64_t>(col0) * kIntermediate + row1] =
                        __float2bfloat16_rn(silu(acc[mi][2]) * acc[mi + 2][2]);
                }
                if (local_col1 < cols) {
                    activation[static_cast<std::int64_t>(col1) * kIntermediate + row0] =
                        __float2bfloat16_rn(silu(acc[mi][1]) * acc[mi + 2][1]);
                    activation[static_cast<std::int64_t>(col1) * kIntermediate + row1] =
                        __float2bfloat16_rn(silu(acc[mi][3]) * acc[mi + 2][3]);
                }
            }
        }
        __syncthreads();
    }
}

struct Q5DownMma {
    static constexpr int kCodeBytes    = Q5RowSplitStorage::kCodeBytesPerGroup;
    static constexpr int kHighBytes    = Q5RowSplitStorage::kHighBytesPerGroup;
    static constexpr int kHighPerChunk = Q5RowSplitStorage::kHighBytesPerChunk;

    __device__ static __forceinline__ void
    decode_eight(unsigned word, const std::uint8_t* high_chunk, float scale, unsigned (&out)[4]) {
        Q5MmaDecodeAtom::decode_eight(word, high_chunk, scale, out);
    }
};

struct Q6DownMma {
    static constexpr int kCodeBytes    = Q6RowSplitStorage::kCodeBytesPerGroup;
    static constexpr int kHighBytes    = Q6RowSplitStorage::kHighBytesPerGroup;
    static constexpr int kHighPerChunk = Q6RowSplitStorage::kHighBytesPerChunk;

    __device__ static __forceinline__ void
    decode_eight(unsigned word, const std::uint8_t* high_chunk, float scale, unsigned (&out)[4]) {
        Q6MmaDecodeAtom::decode_eight(word, high_chunk, scale, out);
    }
};

template <class Codec, int ExpertWarps, int ExpertBN>
__global__ __launch_bounds__(ExpertWarps * 32, 3) void sparse_moe_prefill_qx_down_kernel(
    const __nv_bfloat16* __restrict__ activation, const int* __restrict__ expert_offsets,
    const int* __restrict__ route_job_experts, const int* __restrict__ route_job_columns,
    const int* __restrict__ route_job_count, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ output) {
    constexpr int ExpertThreads = ExpertWarps * 32;
    constexpr int GroupsPerRow  = kIntermediate / 64;
    __shared__ __align__(16) __nv_bfloat16 As[kExpertBM * kExpertBK];
    __shared__ __align__(16) __nv_bfloat16 Bs[kExpertStages][ExpertBN * kExpertBK];
    __shared__ __align__(16) std::uint8_t Cr[kExpertStages][kExpertBM * 32];
    __shared__ __align__(16) std::uint8_t Hr[kExpertStages][kExpertBM * Codec::kHighBytes];
    __shared__ __align__(16) std::uint8_t Sr[kExpertBM * GroupsPerRow * 2];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int a_mat          = lane >> 3;
    const int a_rin          = lane & 7;
    const int a_rowoff       = a_rin + ((a_mat & 1) << 3);
    const int a_coloff       = (a_mat >> 1) << 3;
    const int b_rin          = lane & 7;
    const int b_koff         = ((lane >> 3) & 1) << 3;
    const int gid            = lane >> 2;
    const int lid            = lane & 3;
    constexpr int row_blocks = kHidden / kExpertBM;
    const int total_work     = *route_job_count * row_blocks;
    for (int work = static_cast<int>(blockIdx.x); work < total_work;
         work += static_cast<int>(gridDim.x)) {
        const int route_job   = work / row_blocks;
        const int row_block   = work - route_job * row_blocks;
        const int expert      = route_job_experts[route_job];
        const int row0        = row_block * kExpertBM;
        const int begin       = expert_offsets[expert];
        const int count       = expert_offsets[expert + 1] - begin;
        const int column_base = route_job_columns[route_job];
        const int cols        = count - column_base < ExpertBN ? count - column_base : ExpertBN;
        float acc[4][4]       = {};

        auto stage_scales = [&] {
            for (int row = tid; row < kExpertBM; row += ExpertThreads) {
                const int global_row  = expert * kHidden + row0 + row;
                const std::int64_t gi = static_cast<std::int64_t>(global_row) * GroupsPerRow;
                cp_async<16, Cache::cg>(&Sr[row * GroupsPerRow * 2], &scales[gi * 2]);
            }
        };

        auto stage_inputs = [&](int stage, int kt) {
            const int k0 = kt * kExpertBK;
            for (int item = tid; item < ExpertBN * (kExpertBK / 8); item += ExpertThreads) {
                const int col        = item / (kExpertBK / 8);
                const int k8         = item - col * (kExpertBK / 8);
                const int packed_col = begin + column_base + col;
                auto* dst            = &Bs[stage][col * kExpertBK + gemm_swz64(col, k8 * 8)];
                const auto* src =
                    activation +
                    static_cast<std::int64_t>(col < cols ? packed_col : begin) * kIntermediate +
                    k0 + k8 * 8;
                cp_async_zfill<16, Cache::cg>(dst, src, col < cols ? 16 : 0);
            }

            for (int item = tid; item < kExpertBM * 2; item += ExpertThreads) {
                const int row         = item >> 1;
                const int half        = item & 1;
                const int global_row  = expert * kHidden + row0 + row;
                const std::int64_t gi = static_cast<std::int64_t>(global_row) * GroupsPerRow + kt;
                cp_async<16, Cache::cg>(&Cr[stage][row * 32 + half * 16],
                                        &codes[gi * 32 + half * 16]);
            }
            for (int row = tid; row < kExpertBM; row += ExpertThreads) {
                const int global_row  = expert * kHidden + row0 + row;
                const std::int64_t gi = static_cast<std::int64_t>(global_row) * GroupsPerRow + kt;
                if constexpr (Codec::kHighBytes == 8) {
                    cp_async<8>(&Hr[stage][row * Codec::kHighBytes], &high[gi * Codec::kHighBytes]);
                } else {
                    cp_async<16, Cache::cg>(&Hr[stage][row * Codec::kHighBytes],
                                            &high[gi * Codec::kHighBytes]);
                }
            }
        };

        auto decode_weight = [&](int stage, int kt) {
            constexpr int CodeChunksPerRow = Codec::kCodeBytes / 4;
            constexpr int HighPerChunk     = Codec::kHighPerChunk;
            static_assert(CodeChunksPerRow * 8 == kExpertBK,
                          "a row of codes must decode to exactly the tile's k width");
            for (int item = tid; item < kExpertBM * CodeChunksPerRow; item += ExpertThreads) {
                const int row     = item / CodeChunksPerRow;
                const int chunk   = item - row * CodeChunksPerRow;
                const float scale = __half2float(__ushort_as_half(
                    *reinterpret_cast<const std::uint16_t*>(&Sr[(row * GroupsPerRow + kt) * 2])));
                unsigned decoded[4];
                Codec::decode_eight(
                    *reinterpret_cast<const unsigned*>(&Cr[stage][row * 32 + chunk * 4]),
                    &Hr[stage][row * Codec::kHighBytes + chunk * HighPerChunk], scale, decoded);
                store_vec(&As[row * kExpertBK + gemm_swz64(row, chunk * 8)],
                          make_int4(static_cast<int>(decoded[0]), static_cast<int>(decoded[1]),
                                    static_cast<int>(decoded[2]), static_cast<int>(decoded[3])));
            }
        };

        stage_scales();
        cp_commit();
#pragma unroll
        for (int stage = 0; stage < kExpertStages; ++stage) {
            stage_inputs(stage, stage);
            cp_commit();
        }

#pragma unroll
        for (int kt = 0; kt < kIntermediate / kExpertBK; ++kt) {
            const int stage = kt & 1;
            cp_wait<kExpertStages - 1>();
            __syncthreads();
            decode_weight(stage, kt);
            __syncthreads();

            if (warp * 8 < cols) {
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    unsigned af[4][4];
                    unsigned bf[2];
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                    }
                    const int brow = warp * 8 + b_rin;
                    const int bcol = ki * 16 + b_koff;
                    ldmatrix_x2(bf[0], bf[1],
                                smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        mma_bf16(acc[mi][0], acc[mi][1], acc[mi][2], acc[mi][3], af[mi][0],
                                 af[mi][1], af[mi][2], af[mi][3], bf[0], bf[1]);
                    }
                }
            }

            __syncthreads();
            const int next = kt + kExpertStages;
            if (next < kIntermediate / kExpertBK) { stage_inputs(stage, next); }
            cp_commit();
        }
        cp_wait<0>();
        __syncthreads();

        if (warp * 8 < cols) {
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int output_row0 = row0 + mi * 16 + gid;
                const int output_row1 = output_row0 + 8;
                const int col0        = begin + column_base + warp * 8 + 2 * lid;
                const int col1        = col0 + 1;
                const int local_col0  = warp * 8 + 2 * lid;
                const int local_col1  = local_col0 + 1;
                if (local_col0 < cols) {
                    output[static_cast<std::int64_t>(col0) * kHidden + output_row0] =
                        __float2bfloat16_rn(acc[mi][0]);
                    output[static_cast<std::int64_t>(col0) * kHidden + output_row1] =
                        __float2bfloat16_rn(acc[mi][2]);
                }
                if (local_col1 < cols) {
                    output[static_cast<std::int64_t>(col1) * kHidden + output_row0] =
                        __float2bfloat16_rn(acc[mi][1]);
                    output[static_cast<std::int64_t>(col1) * kHidden + output_row1] =
                        __float2bfloat16_rn(acc[mi][3]);
                }
            }
        }
        __syncthreads();
    }
}

template <bool Routed, bool Adaptive = false>
__global__ __launch_bounds__(kExpertThreads, 1) void sparse_moe_prefill_w8_down_kernel(
    const __nv_bfloat16* __restrict__ input, const int* __restrict__ expert_offsets,
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ grouped_output, const float* __restrict__ routed_sum,
    const float* __restrict__ shared_scale, __nv_bfloat16* __restrict__ destination, int tokens,
    const int* __restrict__ route_job_count) {
    __shared__ __align__(16) __nv_bfloat16 As[kExpertBM * kExpertBK];
    __shared__ __align__(16) __nv_bfloat16 Bs[kExpertStages][kExpertBN * kExpertBK];
    __shared__ __align__(16) std::uint8_t Cr[kExpertBM * kExpertBK];
    __shared__ __align__(16) std::uint8_t Sr[kExpertBM * 16];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if constexpr (Adaptive) {
        if (*route_job_count < 0) { return; }
    }
    const int expert = Routed ? static_cast<int>(blockIdx.y) : 0;
    const int row0   = static_cast<int>(blockIdx.x) * kExpertBM;
    const int begin  = Routed ? expert_offsets[expert] : static_cast<int>(blockIdx.y) * kExpertBN;
    const int count  = Routed ? expert_offsets[expert + 1] - begin
                              : (tokens - begin < kExpertBN ? tokens - begin : kExpertBN);
    if (count <= 0) { return; }

    const int a_mat              = lane >> 3;
    const int a_rin              = lane & 7;
    const int a_rowoff           = a_rin + ((a_mat & 1) << 3);
    const int a_coloff           = (a_mat >> 1) << 3;
    const int b_rin              = lane & 7;
    const int b_koff             = ((lane >> 3) & 1) << 3;
    const int gid                = lane >> 2;
    const int lid                = lane & 3;
    constexpr int groups_per_row = kIntermediate / 32;

    const int column_iterations = Routed ? (count + kExpertBN - 1) / kExpertBN : 1;
    for (int iteration = 0; iteration < column_iterations; ++iteration) {
        const int column_base = iteration * kExpertBN;
        const int cols        = count - column_base < kExpertBN ? count - column_base : kExpertBN;
        float acc[4][4]       = {};

        auto stage_x = [&](int stage, int kt) {
            const int k0 = kt * kExpertBK;
            for (int item = tid; item < kExpertBN * (kExpertBK / 8); item += kExpertThreads) {
                const int col        = item / (kExpertBK / 8);
                const int k8         = item - col * (kExpertBK / 8);
                const int source_col = begin + column_base + col;
                auto* dst            = &Bs[stage][col * kExpertBK + gemm_swz64(col, k8 * 8)];
                const auto* src =
                    input +
                    static_cast<std::int64_t>(col < cols ? source_col : begin) * kIntermediate +
                    k0 + k8 * 8;
                cp_async_zfill<16, Cache::cg>(dst, src, col < cols ? 16 : 0);
            }
        };

        auto stage_weight = [&](int kt) {
            constexpr int groups_per_tile   = kExpertBK / 32;
            constexpr int scale_cache_tiles = 8 / groups_per_tile;
            const int group0                = kt * groups_per_tile;
            for (int item = tid; item < kExpertBM * (kExpertBK / 16); item += kExpertThreads) {
                const int row        = item / (kExpertBK / 16);
                const int chunk      = item - row * (kExpertBK / 16);
                const int global_row = (Routed ? expert * kHidden : 0) + row0 + row;
                const std::int64_t gi =
                    static_cast<std::int64_t>(global_row) * groups_per_row + group0;
                cp_async<16, Cache::cg>(&Cr[row * kExpertBK + chunk * 16],
                                        &codes[gi * 32 + chunk * 16]);
            }
            if ((kt % scale_cache_tiles) == 0) {
                for (int row = tid; row < kExpertBM; row += kExpertThreads) {
                    const int global_row = (Routed ? expert * kHidden : 0) + row0 + row;
                    const std::int64_t gi =
                        static_cast<std::int64_t>(global_row) * groups_per_row + group0;
                    cp_async<16, Cache::cg>(&Sr[row * 16], &scales[gi * 2]);
                }
            }
        };

        auto decode_weight = [&](int kt) {
            constexpr int groups_per_tile   = kExpertBK / 32;
            constexpr int scale_cache_tiles = 8 / groups_per_tile;
            const int scale_offset          = (kt % scale_cache_tiles) * groups_per_tile * 2;
            const int half                  = lane >> 4;
            const int half_lane             = lane & 15;
            for (int row_pair = warp * 2; row_pair < kExpertBM; row_pair += kExpertWarps * 2) {
                const int row = row_pair + half;
                unsigned scale_pair =
                    half_lane == 0
                        ? *reinterpret_cast<const std::uint32_t*>(&Sr[row * 16 + scale_offset])
                        : 0;
                scale_pair = __shfl_sync(0xffffffffu, scale_pair, half * 16);
#pragma unroll
                for (int group = 0; group < groups_per_tile; ++group) {
                    const float scale = __half2float(__ushort_as_half(
                        static_cast<std::uint16_t>((scale_pair >> (group * 16)) & 0xffffu)));
                    const int col     = group * 32 + half_lane * 2;
                    const std::uint16_t packed =
                        *reinterpret_cast<const std::uint16_t*>(&Cr[row * kExpertBK + col]);
                    const int q0 = static_cast<int>(static_cast<std::int8_t>(packed & 0xffu));
                    const int q1 = static_cast<int>(static_cast<std::int8_t>(packed >> 8));
                    const __nv_bfloat162 value = __floats2bfloat162_rn(
                        static_cast<float>(q0) * scale, static_cast<float>(q1) * scale);
                    store_vec(&As[row * kExpertBK + gemm_swz64(row, col)], value);
                }
            }
        };

        stage_x(0, 0);
        stage_weight(0);
        cp_commit();

#pragma unroll
        for (int kt = 0; kt < kIntermediate / kExpertBK; ++kt) {
            const int stage = kt & 1;
            cp_wait<0>();
            __syncthreads();
            decode_weight(kt);
            __syncthreads();

            const int next = kt + 1;
            if (next < kIntermediate / kExpertBK) {
                stage_x(next & 1, next);
                stage_weight(next);
                cp_commit();
            }

            if (warp * 8 < cols) {
                unsigned af[2][4][4];
                unsigned bf[2][2];
                auto load_fragments = [&](int slot, int ki) {
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[slot][mi][0], af[slot][mi][1], af[slot][mi][2],
                                    af[slot][mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                    }
                    const int brow = warp * 8 + b_rin;
                    const int bcol = ki * 16 + b_koff;
                    ldmatrix_x2(bf[slot][0], bf[slot][1],
                                smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
                };
                load_fragments(0, 0);
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    const int slot = ki & 1;
                    if (ki + 1 < kExpertBK / 16) { load_fragments(slot ^ 1, ki + 1); }
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        mma_bf16(acc[mi][0], acc[mi][1], acc[mi][2], acc[mi][3], af[slot][mi][0],
                                 af[slot][mi][1], af[slot][mi][2], af[slot][mi][3], bf[slot][0],
                                 bf[slot][1]);
                    }
                }
            }
        }
        cp_wait<0>();
        __syncthreads();

        if (warp * 8 < cols) {
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int output_row0 = row0 + mi * 16 + gid;
                const int output_row1 = output_row0 + 8;
                const int col0        = begin + column_base + warp * 8 + 2 * lid;
                const int col1        = col0 + 1;
                const int local_col0  = warp * 8 + 2 * lid;
                const int local_col1  = local_col0 + 1;
                auto store            = [&](int col, int row, float value) {
                    if constexpr (Routed) {
                        grouped_output[static_cast<std::int64_t>(col) * kHidden + row] =
                            __float2bfloat16_rn(value);
                    } else {
                        const float merged =
                            __bfloat162float(
                                destination[static_cast<std::int64_t>(col) * kHidden + row]) +
                            routed_sum[static_cast<std::int64_t>(col) * kHidden + row] +
                            shared_scale[col] * value;
                        destination[static_cast<std::int64_t>(col) * kHidden + row] =
                            __float2bfloat16_rn(merged);
                    }
                };
                if (local_col0 < cols) {
                    store(col0, output_row0, acc[mi][0]);
                    store(col0, output_row1, acc[mi][2]);
                }
                if (local_col1 < cols) {
                    store(col1, output_row0, acc[mi][1]);
                    store(col1, output_row1, acc[mi][3]);
                }
            }
        }
        __syncthreads();
    }
}

union alignas(16) SparseMoeBf16x8 {
    uint4 raw;
    __nv_bfloat162 pair[4];
};


#ifdef NINFER_VOLTA_BUILD
// ---------------------------------------------------------------------------
// Volta: prefill GEMMs on tensor cores (mma.sync.m8n8k4 through volta_mma_qk).
//
// The mma expert kernels above trap below sm_80. The first Volta version replaced
// them with weight-stationary SIMT FP32 kernels (~2.3 TFLOP/s effective): the
// routed experts were then 66% of a3b prefill GPU time, the shared expert 6% and
// the router 1% (8K-token prompt, V100-SXM2).
//
// Routed experts (the *_rega kernels): the structure of llama.cpp's Volta
// mmid-hmma kernel. They keep the device-side job list, BN column tiling and
// epilogues of the routing scaffold; each lane owns one weight row and decodes it
// into registers (see the RegA* notes below).
//
// Shared expert and router (dense over every token): a 64-row x 64-k weight tile
// is decoded to FP16 in shared memory, the BF16 activation columns are narrowed
// to FP16, and four warps run volta_mma_qk with weight rows on the 32-row A axis
// and tokens on the 8-row B axis; warp w owns rows [32*(w&1), +32) and columns
// [BN/2*(w>>1), +BN/2). gate/up keep gate and up of the same rows in one lane,
// so SwiGLU is a register epilogue.
//
// Activations are narrowed BF16 -> FP16 (no BF16 mma on sm_70), as every CUTLASS
// Sm70 route in this port does; accumulation is FP32 throughout.
// ---------------------------------------------------------------------------

constexpr int kHmmaThreads = 128;
constexpr int kHmmaRows    = 64;
constexpr int kHmmaK       = 64; // one Q4/Q5/Q6 group per stage
constexpr int kHmmaStride  = kHmmaK + 8; // halves; 144 B rows keep the fragment loads at
                                         // the 8-bank floor (see q4_volta_mma_gemm.cuh)

template <int BN>
__device__ __forceinline__ void hmma_stage_columns(const __nv_bfloat16* __restrict__ src,
                                                   int src_ld, std::int64_t slot0, int cols,
                                                   int k0, __half (*dst)[kHmmaStride]) {
    for (int idx = static_cast<int>(threadIdx.x); idx < BN * (kHmmaK / 8);
         idx += kHmmaThreads) {
        const int c  = idx / (kHmmaK / 8);
        const int kk = (idx % (kHmmaK / 8)) * 8;
        __align__(16) __half v[8];
        if (c < cols) {
            const uint4 raw = load_vec<uint4>(src + (slot0 + c) * src_ld + k0 + kk);
            const auto* b   = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
            for (int e = 0; e < 8; ++e) { v[e] = __float2half_rn(__bfloat162float(b[e])); }
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) { v[e] = __ushort_as_half(0); }
        }
        *reinterpret_cast<uint4*>(&dst[c][kk]) = *reinterpret_cast<const uint4*>(v);
    }
}

__device__ __forceinline__ void hmma_store_eight(const float (&w)[8], __half* dst) {
    __align__(16) __half v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) { v[e] = __float2half_rn(w[e]); }
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(v);
}

// Register-A variants (the structure of llama.cpp's Volta mmid-hmma kernel): each lane owns one
// weight row and decodes that row's codes for a 32-k step straight into the A operand of
// volta_mma_qk (lane = row, half2 x[l] = k 2l, 2l+1 of an 8-k slice), so weights never touch
// shared memory. Raw codes run one 64-k group ahead in registers; the BF16 activation tile is
// narrowed to FP16 into a double-buffered shared tile one step ahead, one barrier per step.
constexpr int kRegAStepK = 32;              // k per step: 4 slices of 8
constexpr int kRegALd    = kRegAStepK + 8;  // shared B tile row stride (halves)

// Raw codes of one row for one 64-k group, plus whatever the codec needs.
struct RegAQ4Raw {
    uint4 lo, hi; // 32 code bytes: k 0..31 and 32..63
    __half scale;
    __device__ __forceinline__ void load(const std::uint8_t* __restrict__ codes,
                                         const std::uint8_t* __restrict__ /*high*/,
                                         const std::uint8_t* __restrict__ scales,
                                         std::int64_t group_index) {
        const uint4* c = reinterpret_cast<const uint4*>(codes + group_index * 32);
        lo    = c[0];
        hi    = c[1];
        scale = __ushort_as_half(reinterpret_cast<const std::uint16_t*>(scales)[group_index]);
    }
    // A slices for step half `h` (0: k 0..31, 1: k 32..63): 4 slices x 4 half2.
    __device__ __forceinline__ void decode(int h, half2 (&a)[4][4]) const {
        const uint4 w        = h == 0 ? lo : hi;
        const std::uint32_t words[4]{w.x, w.y, w.z, w.w};
        const half2 bias     = __half2half2(__ushort_as_half(0x6408)); // 1032
        const half2 sc       = __half2half2(scale);
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const std::uint32_t word = words[s] ^ 0x88888888u;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const std::uint32_t byte = (word >> (8 * j)) & 0xffu;
                const std::uint32_t bits = (byte & 0x0fu) | ((byte & 0xf0u) << 12) | 0x64006400u;
                a[s][j] = __hmul2(__hsub2(half2_from_bits(bits), bias), sc);
            }
        }
    }
};

struct RegAQ5Raw {
    uint4 lo, hi;
    uint2 high; // 8 high-bit bytes: byte c holds bit e of value e of 8-k chunk c
    __half scale;
    __device__ __forceinline__ void load(const std::uint8_t* __restrict__ codes,
                                         const std::uint8_t* __restrict__ highp,
                                         const std::uint8_t* __restrict__ scales,
                                         std::int64_t group_index) {
        const uint4* c = reinterpret_cast<const uint4*>(codes + group_index * 32);
        lo    = c[0];
        hi    = c[1];
        high  = *reinterpret_cast<const uint2*>(highp + group_index * 8);
        scale = __ushort_as_half(reinterpret_cast<const std::uint16_t*>(scales)[group_index]);
    }
    __device__ __forceinline__ void decode(int h, half2 (&a)[4][4]) const {
        const uint4 w        = h == 0 ? lo : hi;
        const std::uint32_t words[4]{w.x, w.y, w.z, w.w};
        const std::uint32_t hb = (h == 0 ? high.x : high.y) ^ 0xffffffffu; // chunks 4h .. 4h+3
        const half2 bias     = __half2half2(__ushort_as_half(0x6410)); // 1040
        const half2 sc       = __half2half2(scale);
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const std::uint32_t hs = (hb >> (8 * s)) & 0xffu;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const std::uint32_t byte = (words[s] >> (8 * j)) & 0xffu;
                std::uint32_t bits = (byte & 0x0fu) | ((byte & 0xf0u) << 12) | 0x64006400u;
                bits |= (((hs >> (2 * j)) & 1u) << 4) | (((hs >> (2 * j + 1)) & 1u) << 20);
                a[s][j] = __hmul2(__hsub2(half2_from_bits(bits), bias), sc);
            }
        }
    }
};

struct RegAQ6Raw {
    uint4 lo, hi;
    uint4 high; // 16 high-bit bytes: u16 c holds 2 bits per value of 8-k chunk c
    __half scale;
    __device__ __forceinline__ void load(const std::uint8_t* __restrict__ codes,
                                         const std::uint8_t* __restrict__ highp,
                                         const std::uint8_t* __restrict__ scales,
                                         std::int64_t group_index) {
        const uint4* c = reinterpret_cast<const uint4*>(codes + group_index * 32);
        lo    = c[0];
        hi    = c[1];
        high  = *reinterpret_cast<const uint4*>(highp + group_index * 16);
        scale = __ushort_as_half(reinterpret_cast<const std::uint16_t*>(scales)[group_index]);
    }
    __device__ __forceinline__ void decode(int h, half2 (&a)[4][4]) const {
        const uint4 w        = h == 0 ? lo : hi;
        const std::uint32_t words[4]{w.x, w.y, w.z, w.w};
        const std::uint32_t hw[2]{h == 0 ? high.x : high.z, h == 0 ? high.y : high.w};
        const half2 bias     = __half2half2(__ushort_as_half(0x6420)); // 1056
        const half2 sc       = __half2half2(scale);
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const std::uint32_t hs = ((hw[s / 2] >> (16 * (s % 2))) & 0xffffu) ^ 0xaaaau;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const std::uint32_t byte = (words[s] >> (8 * j)) & 0xffu;
                std::uint32_t bits = (byte & 0x0fu) | ((byte & 0xf0u) << 12) | 0x64006400u;
                bits |= (((hs >> (4 * j)) & 3u) << 4) | (((hs >> (4 * j + 2)) & 3u) << 20);
                a[s][j] = __hmul2(__hsub2(half2_from_bits(bits), bias), sc);
            }
        }
    }
};

// One step's activation slice: BN tokens x 32 k of BF16, at most one uint4 (8 values) per thread
// per pass; columns past `cols` re-read the last valid token (their results are never stored).
template <int BN, int THREADS>
struct RegABRaw {
    static constexpr int kVecs   = BN * (kRegAStepK / 8);
    static constexpr int kPasses = (kVecs + THREADS - 1) / THREADS;
    uint4 v[kPasses];
    __device__ __forceinline__ void load(const __nv_bfloat16* __restrict__ src, int ld,
                                         std::int64_t slot0, int cols, int k0) {
#pragma unroll
        for (int p = 0; p < kPasses; ++p) {
            const int i = static_cast<int>(threadIdx.x) + p * THREADS;
            if (i < kVecs) {
                const int c = min(i / 4, cols - 1);
                v[p]        = load_vec<uint4>(src + (slot0 + c) * ld + k0 + (i % 4) * 8);
            }
        }
    }
    __device__ __forceinline__ void store(__half* __restrict__ tile) const {
#pragma unroll
        for (int p = 0; p < kPasses; ++p) {
            const int i = static_cast<int>(threadIdx.x) + p * THREADS;
            if (i < kVecs) {
                const auto* b = reinterpret_cast<const __nv_bfloat16*>(&v[p]);
                __align__(16) __half x[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) { x[e] = __float2half_rn(__bfloat162float(b[e])); }
                *reinterpret_cast<uint4*>(tile + (i / 4) * kRegALd + (i % 4) * 8) =
                    *reinterpret_cast<const uint4*>(x);
            }
        }
    }
};

// C[t] += A(32 rows x 32 k) B(8 tokens of tile t x 32 k)^T for all NT token tiles.
template <int NT>
__device__ __forceinline__ void rega_step_mma(float (&C)[NT][8], const half2 (&a)[4][4],
                                              const __half* __restrict__ tile) {
    const int b_i = volta_k_get_i();
#pragma unroll
    for (int s = 0; s < 4; ++s) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            half2 b[4];
            *reinterpret_cast<uint4*>(b) =
                *reinterpret_cast<const uint4*>(tile + (t * 8 + b_i) * kRegALd + 8 * s);
            volta_mma_qk(C[t], a[s], b);
        }
    }
}

// gate/up: 8 warps; warps 0-3 own gate rows row0 + 32*w + lane, warps 4-7 the same up rows.
// The up accumulators go through shared memory to the gate warps for the SwiGLU epilogue.
template <int BN>
__global__ __launch_bounds__(256, 2) void sparse_moe_prefill_q4_gate_up_rega_kernel(
    const __nv_bfloat16* __restrict__ gathered, const int* __restrict__ expert_offsets,
    const int* __restrict__ route_job_experts, const int* __restrict__ route_job_columns,
    const int* __restrict__ route_job_count, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ activation) {
    constexpr int kThreadsGU    = 256;
    constexpr int kRows         = 128;
    constexpr int kRowBlocks    = kIntermediate / kRows;
    constexpr int kExpertRows   = 2 * kIntermediate;
    constexpr int kGroupsPerRow = kHidden / 64;
    constexpr int NT            = BN / 8;
    constexpr int kTile         = BN * kRegALd;

    __shared__ __align__(16) __half tiles[2 * kTile];
    __shared__ float up_out[kRows][BN + 1];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int gu   = warp >> 2;
    const int rw   = warp & 3;

    const int total_work = *route_job_count * kRowBlocks;
    for (int work = static_cast<int>(blockIdx.x); work < total_work;
         work += static_cast<int>(gridDim.x)) {
        const int route_job   = work / kRowBlocks;
        const int row_block   = work - route_job * kRowBlocks;
        const int expert      = route_job_experts[route_job];
        const int begin       = expert_offsets[expert];
        const int count       = expert_offsets[expert + 1] - begin;
        const int column_base = route_job_columns[route_job];
        const int cols        = min(count - column_base, BN);
        if (cols <= 0) { continue; }
        const std::int64_t slot0 = static_cast<std::int64_t>(begin) + column_base;
        const int row_local      = rw * 32 + lane;
        const std::int64_t wrow  = static_cast<std::int64_t>(expert) * kExpertRows +
                                  (gu != 0 ? kIntermediate : 0) + row_block * kRows + row_local;
        const std::int64_t g0 = wrow * kGroupsPerRow;

        float C[NT][8];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int l = 0; l < 8; ++l) { C[t][l] = 0.0f; }
        }

        // (Issuing the next item's loads before the epilogue, as the down kernel does, measured
        // 2x slower here; with K = 2048 the per-item prologue is already amortized.)
        RegAQ4Raw raw_cur, raw_next;
        RegABRaw<BN, kThreadsGU> braw;
        raw_cur.load(codes, nullptr, scales, g0);
        braw.load(gathered, kHidden, slot0, cols, 0);
        __syncthreads(); // the previous item's epilogue may still read up_out / tiles

        for (int g = 0; g < kGroupsPerRow; ++g) {
            if (g + 1 < kGroupsPerRow) { raw_next.load(codes, nullptr, scales, g0 + g + 1); }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int step = 2 * g + h;
                __half* tile   = tiles + (step & 1) * kTile;
                braw.store(tile);
                if (step + 1 < 2 * kGroupsPerRow) {
                    braw.load(gathered, kHidden, slot0, cols, (step + 1) * kRegAStepK);
                }
                half2 a[4][4];
                raw_cur.decode(h, a);
                __syncthreads();
                rega_step_mma<NT>(C, a, tile);
            }
            raw_cur = raw_next;
        }

        if (gu != 0) {
#pragma unroll
            for (int t = 0; t < NT; ++t) {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    up_out[rw * 32 + volta_d_get_i(l)][t * 8 + volta_d_get_j(l)] = C[t][l];
                }
            }
        }
        __syncthreads();
        if (gu == 0) {
#pragma unroll
            for (int t = 0; t < NT; ++t) {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const int r   = rw * 32 + volta_d_get_i(l);
                    const int col = t * 8 + volta_d_get_j(l);
                    if (col < cols) {
                        activation[(slot0 + col) * kIntermediate + row_block * kRows + r] =
                            __float2bfloat16_rn(silu(C[t][l]) * up_out[r][col]);
                    }
                }
            }
        }
    }
}


// down: 4 warps, 128 rows of the [kHidden, kIntermediate] expert matrix per item.
template <class Raw, int BN>
__global__ __launch_bounds__(128, 4) void sparse_moe_prefill_qx_down_rega_kernel(
    const __nv_bfloat16* __restrict__ activation, const int* __restrict__ expert_offsets,
    const int* __restrict__ route_job_experts, const int* __restrict__ route_job_columns,
    const int* __restrict__ route_job_count, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ output) {
    constexpr int kThreadsD     = 128;
    constexpr int kRows         = 128;
    constexpr int kRowBlocks    = kHidden / kRows;
    constexpr int kGroupsPerRow = kIntermediate / 64;
    constexpr int NT            = BN / 8;
    constexpr int kTile         = BN * kRegALd;

    __shared__ __align__(16) __half tiles[2 * kTile];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;

    // Work item -> (job, row block); the first loads of the next item are issued before the
    // current item's epilogue (K is only 512 here, so an exposed prologue would dominate).
    struct Item {
        int work, row_block, cols;
        std::int64_t slot0, g0;
    };
    const int total_work = *route_job_count * kRowBlocks;
    const auto find = [&](int work) -> Item {
        for (; work < total_work; work += static_cast<int>(gridDim.x)) {
            const int route_job   = work / kRowBlocks;
            const int row_block   = work - route_job * kRowBlocks;
            const int expert      = route_job_experts[route_job];
            const int begin       = expert_offsets[expert];
            const int count       = expert_offsets[expert + 1] - begin;
            const int column_base = route_job_columns[route_job];
            const int cols        = min(count - column_base, BN);
            if (cols <= 0) { continue; }
            const int row = row_block * kRows + warp * 32 + lane;
            return {work, row_block, cols, static_cast<std::int64_t>(begin) + column_base,
                    (static_cast<std::int64_t>(expert) * kHidden + row) * kGroupsPerRow};
        }
        return {total_work, 0, 0, 0, 0};
    };

    Raw raw_cur, raw_next;
    RegABRaw<BN, kThreadsD> braw;
    Item it = find(static_cast<int>(blockIdx.x));
    if (it.work < total_work) {
        raw_cur.load(codes, high, scales, it.g0);
        braw.load(activation, kIntermediate, it.slot0, it.cols, 0);
    }
    while (it.work < total_work) {
        float C[NT][8];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int l = 0; l < 8; ++l) { C[t][l] = 0.0f; }
        }
        __syncthreads(); // the previous item may still read the tiles

        for (int g = 0; g < kGroupsPerRow; ++g) {
            if (g + 1 < kGroupsPerRow) { raw_next.load(codes, high, scales, it.g0 + g + 1); }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int step = 2 * g + h;
                __half* tile   = tiles + (step & 1) * kTile;
                braw.store(tile);
                if (step + 1 < 2 * kGroupsPerRow) {
                    braw.load(activation, kIntermediate, it.slot0, it.cols,
                              (step + 1) * kRegAStepK);
                }
                half2 a[4][4];
                raw_cur.decode(h, a);
                __syncthreads();
                rega_step_mma<NT>(C, a, tile);
            }
            if (g + 1 < kGroupsPerRow) { raw_cur = raw_next; }
        }

        const Item done = it;
        it = find(it.work + static_cast<int>(gridDim.x));
        if (it.work < total_work) {
            raw_cur.load(codes, high, scales, it.g0);
            braw.load(activation, kIntermediate, it.slot0, it.cols, 0);
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int r   = done.row_block * kRows + warp * 32 + volta_d_get_i(l);
                const int col = t * 8 + volta_d_get_j(l);
                if (col < done.cols) {
                    output[(done.slot0 + col) * kHidden + r] = __float2bfloat16_rn(C[t][l]);
                }
            }
        }
    }
}

// W8 rowsplit: one int8 code per element with a row's codes contiguous
// (kCodeBytesPerGroup == kGroupK), one FP16 scale per 32-element group.
__device__ __forceinline__ void simt_decode_w8_row(const std::uint8_t* __restrict__ codes,
                                                   const std::uint8_t* __restrict__ scales,
                                                   std::int64_t row, int K, int k0,
                                                   float (&w)[8]) {
    const int groups_per_row       = K / W8RowSplitStorage::kGroupK;
    const std::int64_t group_index = row * groups_per_row + k0 / W8RowSplitStorage::kGroupK;
    const float scale              = __half2float(__ushort_as_half(
        *reinterpret_cast<const std::uint16_t*>(
                         scales + group_index * W8RowSplitStorage::kScaleBytesPerGroup)));
    const auto* c = reinterpret_cast<const std::int8_t*>(codes) + row * K + k0;
#pragma unroll
    for (int e = 0; e < 8; ++e) { w[e] = static_cast<float>(c[e]) * scale; }
}

// Router on tensor cores: scores[col][row] = router[row] . x[col], 64 router rows x 64 tokens per
// CTA (the last row block is partial: 257 = 4 x 64 + 1). BF16 weights and activations are
// narrowed to FP16 (exact in the normal range) and multiplied with FP32 accumulation.
template <int BN>
__global__ __launch_bounds__(kHmmaThreads) void sparse_moe_prefill_router_hmma_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ router,
    float* __restrict__ scores, int tokens) {
    constexpr int kWarpTiles = BN / 16;
    __shared__ __align__(16) __half Ws[kHmmaRows][kHmmaStride];
    __shared__ __align__(16) __half As[BN][kHmmaStride];

    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5;
    const int row_half = warp & 1;
    const int col_base = (warp >> 1) * (BN / 2);

    const int row0        = static_cast<int>(blockIdx.x) * kHmmaRows;
    const int column_base = static_cast<int>(blockIdx.y) * BN;
    const int cols        = min(tokens - column_base, BN);
    if (cols <= 0) { return; }

    float d[kWarpTiles][8] = {};
    for (int k0 = 0; k0 < kHidden; k0 += kHmmaK) {
        __syncthreads();
        hmma_stage_columns<BN>(x, kHidden, column_base, cols, k0, As);
        for (int idx = tid; idx < kHmmaRows * (kHmmaK / 8); idx += kHmmaThreads) {
            const int row_local = idx / (kHmmaK / 8);
            const int kk        = (idx % (kHmmaK / 8)) * 8;
            const int row       = row0 + row_local;
            __align__(16) __half v[8];
            if (row < kRouterRows) {
                const uint4 raw =
                    load_vec<uint4>(router + static_cast<std::int64_t>(row) * kHidden + k0 + kk);
                const auto* b = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
                for (int e = 0; e < 8; ++e) { v[e] = __float2half_rn(__bfloat162float(b[e])); }
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) { v[e] = __ushort_as_half(0); }
            }
            *reinterpret_cast<uint4*>(&Ws[row_local][kk]) = *reinterpret_cast<const uint4*>(v);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < kHmmaK; kk += 8) {
            half2 a[4];
            volta_load_qp(a, reinterpret_cast<const half2*>(&Ws[32 * row_half][kk]),
                          kHmmaStride / 2);
#pragma unroll
            for (int j = 0; j < kWarpTiles; ++j) {
                half2 b[4];
                volta_load_k(b, reinterpret_cast<const half2*>(&As[col_base + 8 * j][kk]),
                             kHmmaStride / 2);
                volta_mma_qk(d[j], a, b);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < kWarpTiles; ++j) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int local_col = col_base + 8 * j + volta_d_get_j(l);
            const int row       = row0 + 32 * row_half + volta_d_get_i(l);
            if (local_col >= cols || row >= kRouterRows) { continue; }
            // Row stride is the padded kSparseMoeRouterScoreRows (260), as in the mma kernel.
            scores[static_cast<std::int64_t>(column_base + local_col) * kSparseMoeRouterScoreRows +
                   row] = d[j][l];
        }
    }
}

// Shared expert on tensor cores: the dense counterparts of the grouped hmma kernels above (one
// column tile per blockIdx.y instead of a job list). gate/up fuses SwiGLU; down fuses the combine
// the mma kernel performs: destination = destination (residual) + routed_sum + shared_scale * value.
template <int BN, bool Adaptive>
__global__ __launch_bounds__(kHmmaThreads) void sparse_moe_prefill_w8_shared_gate_up_hmma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ activation, int tokens,
    const int* __restrict__ route_job_count) {
    if constexpr (Adaptive) {
        if (*route_job_count < 0) { return; }
    }
    constexpr int kWarpTiles = BN / 16;
    static_assert(BN % 16 == 0 && kIntermediate % kHmmaRows == 0);

    __shared__ __align__(16) __half Wg[kHmmaRows][kHmmaStride];
    __shared__ __align__(16) __half Wu[kHmmaRows][kHmmaStride];
    __shared__ __align__(16) __half As[BN][kHmmaStride];

    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5;
    const int row_half = warp & 1;
    const int col_base = (warp >> 1) * (BN / 2);

    const int row0        = static_cast<int>(blockIdx.x) * kHmmaRows;
    const int column_base = static_cast<int>(blockIdx.y) * BN;
    const int cols        = min(tokens - column_base, BN);
    if (cols <= 0) { return; }

    float dg[kWarpTiles][8] = {};
    float du[kWarpTiles][8] = {};

    for (int k0 = 0; k0 < kHidden; k0 += kHmmaK) {
        __syncthreads();
        hmma_stage_columns<BN>(x, kHidden, column_base, cols, k0, As);
        for (int idx = tid; idx < kHmmaRows * 2 * (kHmmaK / 8); idx += kHmmaThreads) {
            const int chunk        = idx % (kHmmaK / 8);
            const int gu           = (idx / (kHmmaK / 8)) % 2;
            const int row_local    = idx / ((kHmmaK / 8) * 2);
            const std::int64_t row = (gu != 0 ? kIntermediate : 0) + row0 + row_local;
            float w[8];
            simt_decode_w8_row(codes, scales, row, kHidden, k0 + chunk * 8, w);
            hmma_store_eight(w, gu != 0 ? &Wu[row_local][chunk * 8] : &Wg[row_local][chunk * 8]);
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kHmmaK; kk += 8) {
            half2 ag[4], au[4];
            volta_load_qp(ag, reinterpret_cast<const half2*>(&Wg[32 * row_half][kk]),
                          kHmmaStride / 2);
            volta_load_qp(au, reinterpret_cast<const half2*>(&Wu[32 * row_half][kk]),
                          kHmmaStride / 2);
#pragma unroll
            for (int j = 0; j < kWarpTiles; ++j) {
                half2 b[4];
                volta_load_k(b, reinterpret_cast<const half2*>(&As[col_base + 8 * j][kk]),
                             kHmmaStride / 2);
                volta_mma_qk(dg[j], ag, b);
                volta_mma_qk(du[j], au, b);
            }
        }
    }

#pragma unroll
    for (int j = 0; j < kWarpTiles; ++j) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int local_col = col_base + 8 * j + volta_d_get_j(l);
            if (local_col >= cols) { continue; }
            const int row          = row0 + 32 * row_half + volta_d_get_i(l);
            const std::int64_t col = column_base + local_col;
            activation[col * kIntermediate + row] = __float2bfloat16_rn(silu(dg[j][l]) * du[j][l]);
        }
    }
}

template <int BN, bool Adaptive>
__global__ __launch_bounds__(kHmmaThreads) void sparse_moe_prefill_w8_shared_down_hmma_kernel(
    const __nv_bfloat16* __restrict__ activation, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, const float* __restrict__ routed_sum,
    const float* __restrict__ shared_scale, __nv_bfloat16* __restrict__ destination, int tokens,
    const int* __restrict__ route_job_count) {
    if constexpr (Adaptive) {
        if (*route_job_count < 0) { return; }
    }
    constexpr int kWarpTiles = BN / 16;
    static_assert(BN % 16 == 0 && kHidden % kHmmaRows == 0);

    __shared__ __align__(16) __half Ws[kHmmaRows][kHmmaStride];
    __shared__ __align__(16) __half As[BN][kHmmaStride];

    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5;
    const int row_half = warp & 1;
    const int col_base = (warp >> 1) * (BN / 2);

    const int row0        = static_cast<int>(blockIdx.x) * kHmmaRows;
    const int column_base = static_cast<int>(blockIdx.y) * BN;
    const int cols        = min(tokens - column_base, BN);
    if (cols <= 0) { return; }

    float d[kWarpTiles][8] = {};

    for (int k0 = 0; k0 < kIntermediate; k0 += kHmmaK) {
        __syncthreads();
        hmma_stage_columns<BN>(activation, kIntermediate, column_base, cols, k0, As);
        for (int idx = tid; idx < kHmmaRows * (kHmmaK / 8); idx += kHmmaThreads) {
            const int chunk     = idx % (kHmmaK / 8);
            const int row_local = idx / (kHmmaK / 8);
            float w[8];
            simt_decode_w8_row(codes, scales, row0 + row_local, kIntermediate, k0 + chunk * 8, w);
            hmma_store_eight(w, &Ws[row_local][chunk * 8]);
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < kHmmaK; kk += 8) {
            half2 a[4];
            volta_load_qp(a, reinterpret_cast<const half2*>(&Ws[32 * row_half][kk]),
                          kHmmaStride / 2);
#pragma unroll
            for (int j = 0; j < kWarpTiles; ++j) {
                half2 b[4];
                volta_load_k(b, reinterpret_cast<const half2*>(&As[col_base + 8 * j][kk]),
                             kHmmaStride / 2);
                volta_mma_qk(d[j], a, b);
            }
        }
    }

#pragma unroll
    for (int j = 0; j < kWarpTiles; ++j) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int local_col = col_base + 8 * j + volta_d_get_j(l);
            if (local_col >= cols) { continue; }
            const int row          = row0 + 32 * row_half + volta_d_get_i(l);
            const std::int64_t col = column_base + local_col;
            const float merged     = __bfloat162float(destination[col * kHidden + row]) +
                                 routed_sum[col * kHidden + row] + shared_scale[col] * d[j][l];
            destination[col * kHidden + row] = __float2bfloat16_rn(merged);
        }
    }
}

#endif // NINFER_VOLTA_BUILD

template <bool Adaptive>
__global__ void sparse_moe_prefill_reduce_kernel(const __nv_bfloat16* __restrict__ grouped_output,
                                                 const int* __restrict__ packed_index,
                                                 const float* __restrict__ alpha,
                                                 float* __restrict__ routed_sum,
                                                 const int* __restrict__ route_job_count) {
    if constexpr (Adaptive) {
        if (*route_job_count < 0) { return; }
    }
    __shared__ int columns[kTopK];
    __shared__ float weights[kTopK];
    const int token = static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);
    if (tid < kTopK) {
        columns[tid] = packed_index[token * kTopK + tid];
        weights[tid] = alpha[token * kTopK + tid];
    }
    __syncthreads();

    float values[8] = {};
#pragma unroll
    for (int route = 0; route < kTopK; ++route) {
        SparseMoeBf16x8 packed;
        packed.raw = load_vec<uint4>(grouped_output +
                                     static_cast<std::int64_t>(columns[route]) * kHidden + tid * 8);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 decoded = __bfloat1622float2(packed.pair[pair]);
            values[2 * pair] += weights[route] * decoded.x;
            values[2 * pair + 1] += weights[route] * decoded.y;
        }
    }
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        routed_sum[static_cast<std::int64_t>(token) * kHidden + tid * 8 + item] = values[item];
    }
}

} // namespace

void sparse_moe_prefill_launch(const Tensor& x, const SparseMoeWeights& weights,
                               Tensor& destination, const SparseMoePrefillPlan& plan,
                               const SparseMoePrefillWorkspace& workspace, cudaStream_t stream) {
    if (x.ne[1] != plan.tokens || destination.ne[1] != plan.tokens || plan.slice_tokens < 1) {
        throw std::invalid_argument("sparse_moe prefill: launch plan does not match tensors");
    }

    const auto* router = static_cast<const __nv_bfloat16*>(weights.router_shared_gate.qdata);
    const auto* routed_gate_codes = static_cast<const std::uint8_t*>(weights.routed_gate_up.qdata);
    const auto* routed_gate_scales =
        static_cast<const std::uint8_t*>(weights.routed_gate_up.scales);
    const auto* routed_down_codes  = static_cast<const std::uint8_t*>(weights.routed_down.qdata);
    const auto* routed_down_high   = static_cast<const std::uint8_t*>(weights.routed_down.qhigh);
    const auto* routed_down_scales = static_cast<const std::uint8_t*>(weights.routed_down.scales);
    const auto* shared_gate_codes  = static_cast<const std::uint8_t*>(weights.shared_gate_up.qdata);
    const auto* shared_gate_scales =
        static_cast<const std::uint8_t*>(weights.shared_gate_up.scales);
    const auto* shared_down_codes  = static_cast<const std::uint8_t*>(weights.shared_down.qdata);
    const auto* shared_down_scales = static_cast<const std::uint8_t*>(weights.shared_down.scales);

    auto* ids               = static_cast<int*>(workspace.token_ids.data);
    auto* alpha             = static_cast<float*>(workspace.token_alpha.data);
    auto* local_rank        = static_cast<int*>(workspace.local_rank.data);
    auto* packed_index      = static_cast<int*>(workspace.packed_index.data);
    auto* shared_scale      = static_cast<float*>(workspace.shared_scale.data);
    auto* tile_counts       = static_cast<int*>(workspace.tile_counts.data);
    auto* tile_bases        = static_cast<int*>(workspace.tile_bases.data);
    auto* offsets           = static_cast<int*>(workspace.expert_offsets.data);
    auto* route_job_experts = static_cast<int*>(workspace.route_job_experts.data);
    auto* route_job_columns = static_cast<int*>(workspace.route_job_columns.data);
    auto* route_job_count   = static_cast<int*>(workspace.route_job_count.data);
    auto* scores            = static_cast<float*>(workspace.score_storage.data);
    auto* shared_activation = static_cast<__nv_bfloat16*>(workspace.shared_activation.data);
    auto* grouped_io        = static_cast<__nv_bfloat16*>(workspace.grouped_io.data);
#ifndef NINFER_VOLTA_BUILD
    auto* packed_token      = static_cast<int*>(workspace.packed_token.data);
#endif
    auto* routed_activation = static_cast<__nv_bfloat16*>(workspace.routed_storage.data);
    auto* routed_sum        = static_cast<float*>(workspace.routed_sum.data);

    for (std::int32_t token0 = 0; token0 < plan.tokens; token0 += plan.slice_tokens) {
        const std::int32_t tokens =
            std::min(plan.slice_tokens, static_cast<std::int32_t>(plan.tokens - token0));
        const Tensor input_slice = x.slice(1, token0, tokens);
        Tensor output_slice      = destination.slice(1, token0, tokens);
        const auto* input        = static_cast<const __nv_bfloat16*>(input_slice.data);
        auto* output             = static_cast<__nv_bfloat16*>(output_slice.data);
        const int route_tiles =
            (tokens + kSparseMoeRouteTileTokens - 1) / kSparseMoeRouteTileTokens;
        const int assignments   = tokens * kTopK;
        const int adaptive_last = weights.routed_down.qtype == QType::Q5G64_F16S   ? 51
                                  : weights.routed_down.qtype == QType::Q6G64_F16S ? 52
                                                                                   : 0;
        const bool adaptive     = tokens >= 47 && tokens <= adaptive_last;

#ifdef NINFER_VOLTA_BUILD
        {
            constexpr int kRouterBN = 64;
            const dim3 grid((kRouterRows + kHmmaRows - 1) / kHmmaRows,
                            (tokens + kRouterBN - 1) / kRouterBN);
            sparse_moe_prefill_router_hmma_kernel<kRouterBN>
                <<<grid, kHmmaThreads, 0, stream>>>(input, router, scores, tokens);
        }
#else
        sparse_moe_prefill_router_mma_kernel<<<dim3((kRouterRows + kRouterBM - 1) / kRouterBM,
                                                    (tokens + kRouterBN - 1) / kRouterBN),
                                               kRouterThreads, 0, stream>>>(input, router, scores,
                                                                            tokens);
#endif // NINFER_VOLTA_BUILD
        CUDA_CHECK(cudaGetLastError());

        sparse_moe_prefill_select_count_kernel<<<route_tiles, kRouterThreads, 0, stream>>>(
            scores, ids, alpha, shared_scale, local_rank, tile_counts, tokens);
        CUDA_CHECK(cudaGetLastError());

#ifdef NINFER_VOLTA_BUILD
        // Eight assignments over 256 experts average one packed column per 32
        // tokens.  BN=64 therefore becomes full only around T=2048; selecting it
        // at the Ampere-family T=768 seam makes the scalar Volta kernels spend
        // about half their FMAs on zero-padded columns at the default T=1024
        // prefill chunk.
        const bool wide_plan = tokens >= 2048;
#else
        const bool wide_plan = tokens >= kSparseMoePrefillWideMin;
#endif // NINFER_VOLTA_BUILD
        const int route_job_bn = wide_plan ? 64 : 32;
        sparse_moe_prefill_scan_kernel<<<1, kExpertThreads, 0, stream>>>(
            tile_counts, tile_bases, offsets, route_job_experts, route_job_columns, route_job_count,
            route_tiles, route_job_bn, tokens, adaptive);
        CUDA_CHECK(cudaGetLastError());

#ifdef NINFER_VOLTA_BUILD
        if (adaptive) {
            auto* adaptive_activations = reinterpret_cast<float*>(grouped_io);
            sparse_moe_decode_launch_d3_small_t(input_slice, weights, ids, adaptive_activations,
                                                tokens, SparseMoeSmallTD3Schedule::Paths3, stream,
                                                route_job_count);
            sparse_moe_decode_launch_d4_small_t(
                weights, output_slice, ids, alpha, shared_scale, adaptive_activations, tokens,
                SparseMoeSmallTD4Schedule::Rows4, stream, route_job_count);
            sparse_moe_prefill_gather_kernel<true><<<assignments, kExpertThreads, 0, stream>>>(
                input, ids, local_rank, packed_index, tile_bases, grouped_io, route_job_count);
        } else {
            sparse_moe_prefill_gather_kernel<false><<<assignments, kExpertThreads, 0, stream>>>(
                input, ids, local_rank, packed_index, tile_bases, grouped_io, nullptr);
        }
#else
        const bool routed_gate_up_q4 = weights.routed_gate_up.qtype == QType::Q4G64_F16S;
        const int index_blocks       = (assignments + kExpertThreads - 1) / kExpertThreads;
        if (adaptive) {
            auto* adaptive_activations = reinterpret_cast<float*>(grouped_io);
            sparse_moe_decode_launch_d3_small_t(input_slice, weights, ids, adaptive_activations,
                                                tokens, SparseMoeSmallTD3Schedule::Paths3, stream,
                                                route_job_count);
            sparse_moe_decode_launch_d4_small_t(
                weights, output_slice, ids, alpha, shared_scale, adaptive_activations, tokens,
                SparseMoeSmallTD4Schedule::Rows4, stream, route_job_count);
            // adaptive implies a Q5/Q6 routed down, and prefill_min_tokens admits those only
            // with a Q4 routed gate/up, so this path is always the indexed one.
            sparse_moe_prefill_index_kernel<true><<<index_blocks, kExpertThreads, 0, stream>>>(
                ids, local_rank, packed_index, tile_bases, packed_token, assignments,
                route_job_count);
        } else if (routed_gate_up_q4) {
            sparse_moe_prefill_index_kernel<false><<<index_blocks, kExpertThreads, 0, stream>>>(
                ids, local_rank, packed_index, tile_bases, packed_token, assignments, nullptr);
        } else {
            sparse_moe_prefill_gather_kernel<false><<<assignments, kExpertThreads, 0, stream>>>(
                input, ids, local_rank, packed_index, tile_bases, grouped_io, nullptr);
        }
#endif
        CUDA_CHECK(cudaGetLastError());

#ifdef NINFER_VOLTA_BUILD
        // The grouped tensor-core kernels take the same device-side job list and the
        // same BN as the scan was told to use, so the column tiling matches exactly.
        if (weights.routed_gate_up.qtype == QType::Q4G64_F16S) {
            if (wide_plan) {
                sparse_moe_prefill_q4_gate_up_rega_kernel<64>
                    <<<kPrefillPersistentBlocks, 256, 0, stream>>>(
                        grouped_io, offsets, route_job_experts, route_job_columns, route_job_count,
                        routed_gate_codes, routed_gate_scales, routed_activation);
            } else {
                sparse_moe_prefill_q4_gate_up_rega_kernel<32>
                    <<<kPrefillPersistentBlocks, 256, 0, stream>>>(
                        grouped_io, offsets, route_job_experts, route_job_columns, route_job_count,
                        routed_gate_codes, routed_gate_scales, routed_activation);
            }
        } else {
            throw std::invalid_argument(
                "sparse_moe prefill: Volta supports only the Q4 routed gate/up codec");
        }
#else
        const dim3 routed_gate_grid(kIntermediate / (kExpertBM / 2), kExperts);
        // Same predicate that decided whether packed_token was written above.
        if (routed_gate_up_q4) {
            if (wide_plan) {
                sparse_moe_prefill_q4_gate_up_kernel<8, 64>
                    <<<kPrefillPersistentBlocks, 8 * 32, 0, stream>>>(
                        input, packed_token, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_gate_codes, routed_gate_scales, routed_activation);
            } else {
                sparse_moe_prefill_q4_gate_up_kernel<4, 32>
                    <<<kPrefillPersistentBlocks, 4 * 32, 0, stream>>>(
                        input, packed_token, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_gate_codes, routed_gate_scales, routed_activation);
            }
        } else if (weights.routed_gate_up.qtype == QType::W8G32_F16S) {
            sparse_moe_prefill_w8_gate_up_kernel<true>
                <<<routed_gate_grid, kExpertThreads, 0, stream>>>(
                    grouped_io, offsets, routed_gate_codes, routed_gate_scales, routed_activation,
                    tokens, nullptr);
        } else {
            throw std::invalid_argument("sparse_moe prefill: unsupported gate/up codec");
        }
#endif // NINFER_VOLTA_BUILD
        CUDA_CHECK(cudaGetLastError());

        const dim3 shared_gate_grid(kIntermediate / (kExpertBM / 2),
                                    (tokens + kExpertBN - 1) / kExpertBN);
#ifdef NINFER_VOLTA_BUILD
        {
            constexpr int kSharedBN = 64;
            const dim3 grid(kIntermediate / kHmmaRows, (tokens + kSharedBN - 1) / kSharedBN);
            if (adaptive) {
                sparse_moe_prefill_w8_shared_gate_up_hmma_kernel<kSharedBN, true>
                    <<<grid, kHmmaThreads, 0, stream>>>(input, shared_gate_codes,
                                                        shared_gate_scales, shared_activation,
                                                        tokens, route_job_count);
            } else {
                sparse_moe_prefill_w8_shared_gate_up_hmma_kernel<kSharedBN, false>
                    <<<grid, kHmmaThreads, 0, stream>>>(input, shared_gate_codes,
                                                        shared_gate_scales, shared_activation,
                                                        tokens, nullptr);
            }
        }
#else
        if (adaptive) {
            sparse_moe_prefill_w8_gate_up_kernel<false, true>
                <<<shared_gate_grid, kExpertThreads, 0, stream>>>(
                    input, nullptr, shared_gate_codes, shared_gate_scales, shared_activation,
                    tokens, route_job_count);
        } else {
            sparse_moe_prefill_w8_gate_up_kernel<false, false>
                <<<shared_gate_grid, kExpertThreads, 0, stream>>>(
                    input, nullptr, shared_gate_codes, shared_gate_scales, shared_activation,
                    tokens, nullptr);
        }
#endif // NINFER_VOLTA_BUILD
        CUDA_CHECK(cudaGetLastError());

#ifdef NINFER_VOLTA_BUILD
        switch (weights.routed_down.qtype) {
        case QType::Q5G64_F16S:
            if (wide_plan) {
                sparse_moe_prefill_qx_down_rega_kernel<RegAQ5Raw, 64>
                    <<<kPrefillPersistentBlocks, 128, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_down_codes, routed_down_high, routed_down_scales,
                        grouped_io);
            } else {
                sparse_moe_prefill_qx_down_rega_kernel<RegAQ5Raw, 32>
                    <<<kPrefillPersistentBlocks, 128, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_down_codes, routed_down_high, routed_down_scales,
                        grouped_io);
            }
            break;
        case QType::Q6G64_F16S:
            if (wide_plan) {
                sparse_moe_prefill_qx_down_rega_kernel<RegAQ6Raw, 64>
                    <<<kPrefillPersistentBlocks, 128, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_down_codes, routed_down_high, routed_down_scales,
                        grouped_io);
            } else {
                sparse_moe_prefill_qx_down_rega_kernel<RegAQ6Raw, 32>
                    <<<kPrefillPersistentBlocks, 128, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_down_codes, routed_down_high, routed_down_scales,
                        grouped_io);
            }
            break;
        default:
            throw std::invalid_argument(
                "sparse_moe prefill: Volta supports only the Q5/Q6 routed down codecs");
        }
#else
        const dim3 routed_down_grid(kHidden / kExpertBM, kExperts);
        switch (weights.routed_down.qtype) {
        case QType::Q5G64_F16S:
            if (wide_plan) {
                sparse_moe_prefill_qx_down_kernel<Q5DownMma, 8, 64>
                    <<<kPrefillPersistentBlocks, 8 * 32, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_down_codes, routed_down_high, routed_down_scales,
                        grouped_io);
            } else {
                sparse_moe_prefill_qx_down_kernel<Q5DownMma, 4, 32>
                    <<<kPrefillPersistentBlocks, 4 * 32, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_down_codes, routed_down_high, routed_down_scales,
                        grouped_io);
            }
            break;
        case QType::Q6G64_F16S:
            if (wide_plan) {
                sparse_moe_prefill_qx_down_kernel<Q6DownMma, 8, 64>
                    <<<kPrefillPersistentBlocks, 8 * 32, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_down_codes, routed_down_high, routed_down_scales,
                        grouped_io);
            } else {
                sparse_moe_prefill_qx_down_kernel<Q6DownMma, 4, 32>
                    <<<kPrefillPersistentBlocks, 4 * 32, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_down_codes, routed_down_high, routed_down_scales,
                        grouped_io);
            }
            break;
        case QType::W8G32_F16S:
            sparse_moe_prefill_w8_down_kernel<true>
                <<<routed_down_grid, kExpertThreads, 0, stream>>>(
                    routed_activation, offsets, routed_down_codes, routed_down_scales, grouped_io,
                    nullptr, nullptr, nullptr, tokens, nullptr);
            break;
        default:
            throw std::invalid_argument("sparse_moe prefill: unsupported down codec");
        }
#endif // NINFER_VOLTA_BUILD
        CUDA_CHECK(cudaGetLastError());

        if (adaptive) {
            sparse_moe_prefill_reduce_kernel<true><<<tokens, kExpertThreads, 0, stream>>>(
                grouped_io, packed_index, alpha, routed_sum, route_job_count);
        } else {
            sparse_moe_prefill_reduce_kernel<false><<<tokens, kExpertThreads, 0, stream>>>(
                grouped_io, packed_index, alpha, routed_sum, nullptr);
        }
        CUDA_CHECK(cudaGetLastError());

        const dim3 shared_down_grid(kHidden / kExpertBM, (tokens + kExpertBN - 1) / kExpertBN);
#ifdef NINFER_VOLTA_BUILD
        {
            constexpr int kSharedBN = 64;
            const dim3 grid(kHidden / kHmmaRows, (tokens + kSharedBN - 1) / kSharedBN);
            if (adaptive) {
                sparse_moe_prefill_w8_shared_down_hmma_kernel<kSharedBN, true>
                    <<<grid, kHmmaThreads, 0, stream>>>(shared_activation, shared_down_codes,
                                                        shared_down_scales, routed_sum,
                                                        shared_scale, output, tokens,
                                                        route_job_count);
            } else {
                sparse_moe_prefill_w8_shared_down_hmma_kernel<kSharedBN, false>
                    <<<grid, kHmmaThreads, 0, stream>>>(shared_activation, shared_down_codes,
                                                        shared_down_scales, routed_sum,
                                                        shared_scale, output, tokens, nullptr);
            }
        }
#else
        if (adaptive) {
            sparse_moe_prefill_w8_down_kernel<false, true>
                <<<shared_down_grid, kExpertThreads, 0, stream>>>(
                    shared_activation, nullptr, shared_down_codes, shared_down_scales, nullptr,
                    routed_sum, shared_scale, output, tokens, route_job_count);
        } else {
            sparse_moe_prefill_w8_down_kernel<false, false>
                <<<shared_down_grid, kExpertThreads, 0, stream>>>(
                    shared_activation, nullptr, shared_down_codes, shared_down_scales, nullptr,
                    routed_sum, shared_scale, output, tokens, nullptr);
        }
#endif // NINFER_VOLTA_BUILD
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
