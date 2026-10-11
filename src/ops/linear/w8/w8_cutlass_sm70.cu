#include "ops/linear/w8/w8_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/linear/w8/w8_rowsplit_storage.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr int kCodesPerThread = 8; // one 8-byte code load -> one 16-byte FP16 store

// W8G32_F16S RowSplit, unpadded in K: row r's codes are k contiguous int8 values and its scales
// k/32 contiguous FP16 values. Each thread decodes eight codes of one group (32 / 8 = 4 threads
// share a scale). Same arithmetic as W8ScalarDecodeAtom: float(code) * float(scale), rounded once
// to FP16.
__global__ void dequant_w8_rows_to_fp16(const std::uint8_t* __restrict__ codes,
                                        const std::uint8_t* __restrict__ scales,
                                        std::int64_t elements, int k,
                                        cutlass::half_t* __restrict__ out) {
    const std::int64_t first =
        (static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kCodesPerThread;
    if (first >= elements) { return; }
    const std::int64_t row   = first / k;
    const int col            = static_cast<int>(first - row * k);
    const std::int64_t group = row * (k / W8RowSplitStorage::kGroupK) +
                               col / W8RowSplitStorage::kGroupK;
    const float scale        = __half2float(
        __ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scales + group * 2)));

    const uint2 packed = *reinterpret_cast<const uint2*>(codes + first);
    const std::uint32_t words[2]{packed.x, packed.y};
    __align__(16) __half values[kCodesPerThread];
#pragma unroll
    for (int i = 0; i < kCodesPerThread; ++i) {
        const auto code = static_cast<std::int8_t>((words[i / 4] >> (8 * (i % 4))) & 0xffU);
        values[i]       = __float2half_rn(static_cast<float>(code) * scale);
    }
    *reinterpret_cast<uint4*>(out + first) = *reinterpret_cast<const uint4*>(values);
}

__global__ void bf16_to_fp16_kernel(const __nv_bfloat16* __restrict__ in,
                                    cutlass::half_t* __restrict__ out, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = cutlass::half_t(__bfloat162float(in[i])); }
}

using ElementAccumulator     = float;
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA          = cutlass::half_t;
using ElementInputB          = cutlass::half_t;
using ElementOutput          = cutlass::bfloat16_t;

using LayoutInputA = cutlass::layout::RowMajor;    // x: [T][k], k contiguous
using LayoutInputB = cutlass::layout::ColumnMajor; // W: [k][n] stored n-major, k contiguous
using LayoutOutput = cutlass::layout::RowMajor;    // out: [T][n], n contiguous

using MMAOp  = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm70;

using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
using ShapeMMAWarp        = cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeMMAOp          = cutlass::gemm::GemmShape<8, 8, 4>;

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using EpilogueOp =
    cutlass::epilogue::thread::LinearCombination<ElementOutput,
                                                 128 / cutlass::sizeof_bits<ElementOutput>::value,
                                                 ElementAccumulator, ElementComputeEpilogue>;

constexpr int kNumStages = 2;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA, LayoutInputA, ElementInputB, LayoutInputB,
                                         ElementOutput, LayoutOutput, ElementAccumulator, MMAOp,
                                         SmArch, ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp,
                                         EpilogueOp, SwizzleThreadBlock, kNumStages>;

typename Gemm::Arguments make_arguments(const cutlass::half_t* x_fp16,
                                        const cutlass::half_t* w_fp16, ElementOutput* out,
                                        std::int32_t out_ld, std::int32_t n, std::int32_t k,
                                        std::int32_t cols, bool accumulate) {
    const ElementComputeEpilogue beta = accumulate ? ElementComputeEpilogue(1)
                                                   : ElementComputeEpilogue(0);
    return typename Gemm::Arguments{cutlass::gemm::GemmCoord(cols, n, k),
                                    {x_fp16, k},
                                    {w_fp16, k},
                                    {out, out_ld},
                                    {out, out_ld},
                                    {ElementComputeEpilogue(1), beta},
                                    1};
}

std::size_t gemm_workspace_bytes_for(std::int32_t n, std::int32_t k, std::int32_t cols,
                                     bool accumulate) {
    return Gemm::get_workspace_size(
        make_arguments(nullptr, nullptr, nullptr, n, n, k, cols, accumulate));
}

struct Scratch {
    Tensor w_fp16;
    Tensor x_fp16;
    DeviceSpan gemm_workspace;
};

template <class Allocator>
Scratch allocate_scratch(Allocator& allocator, std::int32_t parent_rows, std::int32_t k,
                         std::int32_t cols, std::size_t gemm_workspace_bytes) {
    Scratch out;
    out.w_fp16 = allocator.alloc(DType::FP16, {k, parent_rows});
    out.x_fp16 = allocator.alloc(DType::FP16, {k, cols});
    if (gemm_workspace_bytes > 0) {
        out.gemm_workspace = allocator.alloc_bytes(gemm_workspace_bytes);
    }
    return out;
}

std::int32_t elem_ld(const Tensor& t) {
    return static_cast<std::int32_t>(t.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)));
}

} // namespace

std::size_t w8_cutlass_sm70_workspace_bytes(std::int32_t parent_rows, std::int32_t k,
                                            std::int32_t cols, std::int32_t max_output_rows,
                                            bool accumulate) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_scratch(layout, parent_rows, k, cols,
                           gemm_workspace_bytes_for(max_output_rows, k, cols, accumulate));
    return layout.peak_bytes(1);
}

void w8_cutlass_sm70_launch(const Tensor& x, const Weight& parent,
                            const W8CutlassSm70Output* outputs, int output_count, bool accumulate,
                            WorkspaceArena& ws, cudaStream_t stream) {
    const std::int32_t k    = x.ne[0];
    const std::int32_t cols = x.ne[1];
    const std::int32_t n    = parent.n;
    // The dense dequant indexes codes at row*k, which matches the packed layout only when the
    // parent is unpadded in K; the vector loads need whole 8-code runs.
    if (parent.qtype != QType::W8G32_F16S || parent.layout != QuantLayout::RowSplit ||
        parent.k != k || parent.padded_shape[1] != k || k % W8RowSplitStorage::kGroupK != 0 ||
        output_count <= 0) {
        throw std::invalid_argument("w8_cutlass_sm70: unsupported parent weight or problem");
    }
    std::int32_t max_rows = 0;
    for (int i = 0; i < output_count; ++i) {
        const W8CutlassSm70Output& o = outputs[i];
        if (o.out == nullptr || o.row_begin < 0 || o.rows <= 0 || o.row_begin + o.rows > n ||
            o.out->ne[0] != o.rows || o.out->ne[1] != cols) {
            throw std::invalid_argument("w8_cutlass_sm70: invalid output row range");
        }
        max_rows = o.rows > max_rows ? o.rows : max_rows;
    }

    auto scratch_scope = ws.scope();
    const Scratch scratch =
        allocate_scratch(ws, n, k, cols, gemm_workspace_bytes_for(max_rows, k, cols, accumulate));
    auto* w_fp16 = static_cast<cutlass::half_t*>(scratch.w_fp16.data);
    auto* x_fp16 = static_cast<cutlass::half_t*>(scratch.x_fp16.data);

    {
        const std::int64_t elements = static_cast<std::int64_t>(n) * k;
        const int threads           = 256;
        const auto blocks = static_cast<unsigned>(
            (elements / kCodesPerThread + threads - 1) / threads);
        dequant_w8_rows_to_fp16<<<blocks, threads, 0, stream>>>(
            static_cast<const std::uint8_t*>(parent.qdata),
            static_cast<const std::uint8_t*>(parent.scales), elements, k, w_fp16);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        const std::int64_t count = static_cast<std::int64_t>(cols) * k;
        const int threads        = 256;
        const auto blocks        = static_cast<unsigned>((count + threads - 1) / threads);
        bf16_to_fp16_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), x_fp16, count);
        CUDA_CHECK(cudaGetLastError());
    }

    for (int i = 0; i < output_count; ++i) {
        const W8CutlassSm70Output& o = outputs[i];
        Gemm gemm_op;
        const typename Gemm::Arguments arguments = make_arguments(
            x_fp16, w_fp16 + static_cast<std::int64_t>(o.row_begin) * k,
            static_cast<ElementOutput*>(o.out->data), elem_ld(*o.out), o.rows, k, cols,
            accumulate);
        cutlass::Status status = gemm_op.can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("w8_cutlass_sm70: CUTLASS can_implement failed");
        }
        status = gemm_op.initialize(arguments, scratch.gemm_workspace.data, stream);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("w8_cutlass_sm70: CUTLASS initialize failed");
        }
        status = gemm_op(stream);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("w8_cutlass_sm70: CUTLASS gemm() failed");
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
