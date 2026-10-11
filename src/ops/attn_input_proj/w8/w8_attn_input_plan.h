#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class W8AttnInputScheduleId {
    DecodeR8Direct,
    SplitKMmaDirect,
    SimtR8C4,
    MmaR32C64,
    MmaR32C128,
    MmaR64C64,
    MmaR64C96,
    MmaR64C128,
    MmaR128C64,
    MmaR128C80,
    Dflash2SimtSplit, // DFlash2 draft QKV W8 [6144,5120] direct split output (sm_70)
    CutlassSm70,      // Volta wide T, four-output form: FP16 dequant + CUTLASS Sm70 GEMMs
};

struct W8AttnInputProblem {
    std::int32_t input_rows;
    std::int32_t query_rows;
    std::int32_t kv_rows;
    std::int32_t parent_rows;
    std::int32_t padded_k;
    std::int32_t cols;
};

struct W8AttnInputPlan {
    W8AttnInputScheduleId schedule;
};

const char* w8_attn_input_schedule_name(W8AttnInputScheduleId schedule) noexcept;
bool w8_attn_input_admits(const W8AttnInputProblem& problem) noexcept;
W8AttnInputPlan w8_attn_input_resolve_plan(const W8AttnInputProblem& problem);
// Exact transient bytes the resolved four-output schedule takes when given an arena.
[[nodiscard]] std::size_t w8_attn_input_workspace_bytes(const W8AttnInputProblem& problem);

void w8_attn_input_execute_plan(const W8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void w8_attn_input_execute_plan(const W8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& k, Tensor& v, cudaStream_t stream);
// `workspace` may be null (the Op's no-workspace overload): CutlassSm70 then degrades to the
// in-place SIMT route, as gdn_input_proj's CutlassSm70 does.
void w8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                            Tensor& k, Tensor& v, WorkspaceArena* workspace, cudaStream_t stream);
void w8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                            cudaStream_t stream);

} // namespace ninfer::ops::detail
