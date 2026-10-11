#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Volta (sm_70) tensor-core route for wide-T W8G32_F16S RowSplit projections.
//
// On V100 every W8 projection without a Volta mma route runs the SIMT row-split GEMM at
// ~3.3-4.5 TFLOP/s; in the 35B-A3B prefill the attention input projection and the 40 output
// projections (linear_add) were 19% of GPU time on that kernel. Same two-phase treatment as
// gdn_input_proj's CutlassSm70 route: dequantize the parent into an FP16 scratch buffer, cast the
// BF16 activations to FP16, then run CUTLASS's Sm70 tensor-core GEMM once per output row range.
//
// One parent, several outputs: output i receives parent rows [row_begin, row_begin + rows) and is
// a contiguous BF16 [rows, T] tensor (its own row stride). With `accumulate` the epilogue reads
// the output as C and writes C + W x back (beta = 1, FP32 epilogue), the linear_add contract.
struct W8CutlassSm70Output {
    Tensor* out;
    std::int32_t row_begin;
    std::int32_t rows;
};

// Exact transient bytes w8_cutlass_sm70_launch() takes from the arena for this problem: the
// FP16 parent [parent_rows, k], the FP16 activation [k, cols] and CUTLASS's own workspace for the
// largest output.
[[nodiscard]] std::size_t w8_cutlass_sm70_workspace_bytes(std::int32_t parent_rows,
                                                          std::int32_t k, std::int32_t cols,
                                                          std::int32_t max_output_rows,
                                                          bool accumulate);

void w8_cutlass_sm70_launch(const Tensor& x, const Weight& parent,
                            const W8CutlassSm70Output* outputs, int output_count, bool accumulate,
                            WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops::detail
