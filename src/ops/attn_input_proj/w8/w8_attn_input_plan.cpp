#include "ops/attn_input_proj/w8/w8_attn_input_plan.h"

#include "ops/attn_input_proj/w8/w8_attn_input_kernels.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/w8/w8_cutlass_sm70.h"
#endif

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    W8AttnInputScheduleId schedule;
};

constexpr std::array<RouteSpec, 4> kTargetRoutes{{
    {1, 1, W8AttnInputScheduleId::DecodeR8Direct},
    {2, 64, W8AttnInputScheduleId::SplitKMmaDirect},
    {65, 128, W8AttnInputScheduleId::MmaR32C128},
    {129, kAnyCols, W8AttnInputScheduleId::MmaR64C128},
}};

constexpr std::array<RouteSpec, 9> kCompanionRoutes{{
    {1, 1, W8AttnInputScheduleId::DecodeR8Direct},
    {2, 96, W8AttnInputScheduleId::SplitKMmaDirect},
    {97, 192, W8AttnInputScheduleId::MmaR32C64},
    {193, 288, W8AttnInputScheduleId::MmaR64C96},
    {289, 320, W8AttnInputScheduleId::MmaR64C64},
    {321, 384, W8AttnInputScheduleId::MmaR64C128},
    {385, 448, W8AttnInputScheduleId::MmaR128C64},
    {449, 560, W8AttnInputScheduleId::MmaR128C80},
    {561, kAnyCols, W8AttnInputScheduleId::MmaR64C128},
}};

template <std::size_t N>
constexpr bool catalog_is_closed(const std::array<RouteSpec, N>& routes) {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(kTargetRoutes),
              "W8 target attention input routes must be exact and closed");
static_assert(catalog_is_closed(kCompanionRoutes),
              "W8 companion attention input routes must be exact and closed");

bool is_dflash2_shape(const W8AttnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.query_rows == 4096 && problem.kv_rows == 1024 &&
           problem.parent_rows == 6144 && problem.padded_k == 5120;
}

bool is_companion_shape(const W8AttnInputProblem& problem) noexcept {
    return problem.input_rows == 2048 && problem.query_rows == 4096 && problem.kv_rows == 1024 &&
           problem.parent_rows == 6144 && problem.padded_k == 2048;
}

bool supported_shape(const W8AttnInputProblem& problem) noexcept {
    const bool target_qkgv =
        problem.query_rows == 4096 && problem.kv_rows == 512 && problem.parent_rows == 9216;
    if (is_dflash2_shape(problem)) { return true; }
    return problem.input_rows == 2048 && problem.padded_k == 2048 &&
           (target_qkgv || is_companion_shape(problem));
}

} // namespace

const char* w8_attn_input_schedule_name(W8AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case W8AttnInputScheduleId::DecodeR8Direct:
        return "attn_input_proj.w8.decode.r8.direct.k2048";
    case W8AttnInputScheduleId::SplitKMmaDirect:
        return "attn_input_proj.w8.splitk.mma.r16.direct";
    case W8AttnInputScheduleId::SimtR8C4:
        return "attn_input_proj.w8.simt.r8.c4";
    case W8AttnInputScheduleId::MmaR32C64:
        return "attn_input_proj.w8.mma.r32.c64";
    case W8AttnInputScheduleId::MmaR32C128:
        return "attn_input_proj.w8.mma.r32.c128";
    case W8AttnInputScheduleId::MmaR64C64:
        return "attn_input_proj.w8.mma.r64.c64";
    case W8AttnInputScheduleId::MmaR64C96:
        return "attn_input_proj.w8.mma.r64.c96";
    case W8AttnInputScheduleId::MmaR64C128:
        return "attn_input_proj.w8.mma.r64.c128";
    case W8AttnInputScheduleId::MmaR128C64:
        return "attn_input_proj.w8.mma.r128.c64";
    case W8AttnInputScheduleId::MmaR128C80:
        return "attn_input_proj.w8.mma.r128.c80";
    case W8AttnInputScheduleId::Dflash2SimtSplit:
        return "attn_input_proj.w8.dflash2.simt.split.k5120";
    case W8AttnInputScheduleId::CutlassSm70:
        return "attn_input_proj.w8.cutlass.sm70";
    }
    return "attn_input_proj.w8.unknown";
}

bool w8_attn_input_admits(const W8AttnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols > 0;
}

#ifdef NINFER_VOLTA_BUILD
// Lower edge of the CUTLASS route; below it the fixed cost of dequantizing the whole parent is
// not amortized. Measured with ninfer_attn_input_proj_bench; see docs/v100.md,
// "35B-A3B prefill on tensor cores".
constexpr std::int32_t kVoltaCutlassFirstCols = 32;
#endif

W8AttnInputPlan w8_attn_input_resolve_plan(const W8AttnInputProblem& problem) {
    if (!w8_attn_input_admits(problem)) {
        throw std::invalid_argument(
            "W8 attention input: exact problem or column count is not admitted");
    }
#ifdef NINFER_VOLTA_BUILD
    if (is_dflash2_shape(problem)) { return {W8AttnInputScheduleId::Dflash2SimtSplit}; }
    // Same shape as the w8 linear_add override: every schedule in these tables
    // except SimtR8C4 and DecodeR8Direct reaches the trap-stubbed mma/split-K
    // kernels below sm_80, and SimtR8C4's grid derives its token extent from
    // x.ne[1] at launch, so it already covers arbitrary T.
    // T=1 keeps DecodeR8Direct: it is already SIMT and is the decode schedule
    // this port tuned for the 27B target, which shares this Op.
    if (problem.cols == 1) { return {W8AttnInputScheduleId::DecodeR8Direct}; }
    // Wide T on the four-output [9216,2048] parent goes to tensor cores (w8_cutlass_sm70.h):
    // SimtR8C4 runs at ~3.3 TFLOP/s there. The three-output companion keeps SimtR8C4.
    if (problem.cols >= kVoltaCutlassFirstCols && problem.parent_rows == 9216) {
        return {W8AttnInputScheduleId::CutlassSm70};
    }
    return {W8AttnInputScheduleId::SimtR8C4};
#else
    const auto resolve_from = [&](const auto& routes) -> W8AttnInputPlan {
        for (const RouteSpec& route : routes) {
            if (problem.cols >= route.first && problem.cols <= route.last) {
                return {route.schedule};
            }
        }
        throw std::logic_error("W8 attention input: admitted problem has no covering route");
    };
    if (is_companion_shape(problem)) { return resolve_from(kCompanionRoutes); }
    return resolve_from(kTargetRoutes);
#endif // NINFER_VOLTA_BUILD
}

std::size_t w8_attn_input_workspace_bytes(const W8AttnInputProblem& problem) {
    const W8AttnInputPlan plan = w8_attn_input_resolve_plan(problem);
#ifdef NINFER_VOLTA_BUILD
    if (plan.schedule == W8AttnInputScheduleId::CutlassSm70) {
        return w8_cutlass_sm70_workspace_bytes(problem.parent_rows, problem.padded_k, problem.cols,
                                               problem.query_rows, /*accumulate=*/false);
    }
#endif
    (void)plan;
    return 0;
}

void w8_attn_input_execute_plan(const W8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                cudaStream_t stream) {
    const W8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    const W8AttnInputPlan resolved = w8_attn_input_resolve_plan(problem);
    if (problem.parent_rows != 9216 || problem.kv_rows != 512 ||
        resolved.schedule != plan.schedule) {
        throw std::invalid_argument(
            "W8 attention input: plan does not match exact four-output problem");
    }
    switch (plan.schedule) {
    case W8AttnInputScheduleId::CutlassSm70:
        // Needs an arena; w8_attn_input_dispatch() routes it before reaching this table.
        throw std::logic_error("W8 attention input: CutlassSm70 needs a workspace");
    case W8AttnInputScheduleId::DecodeR8Direct:
        w8_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        return;
    case W8AttnInputScheduleId::SplitKMmaDirect:
        w8_attn_input_splitk_mma_launch(x, weight, q, gate, k, v, stream);
        return;
    case W8AttnInputScheduleId::SimtR8C4:
        w8_attn_input_simt_r8_c4_launch(x, weight, q, gate, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR32C128:
        w8_attn_input_mma_r32_c128_launch(x, weight, q, gate, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR64C128:
        w8_attn_input_mma_r64_c128_launch(x, weight, q, gate, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR32C64:
    case W8AttnInputScheduleId::MmaR64C64:
    case W8AttnInputScheduleId::MmaR64C96:
    case W8AttnInputScheduleId::MmaR128C64:
    case W8AttnInputScheduleId::MmaR128C80:
    case W8AttnInputScheduleId::Dflash2SimtSplit:
        throw std::logic_error("W8 attention input: schedule not valid for this output form");
    }
    throw std::logic_error("W8 attention input: unknown schedule");
}

void w8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                            Tensor& k, Tensor& v, WorkspaceArena* workspace, cudaStream_t stream) {
    const W8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    const W8AttnInputPlan plan = w8_attn_input_resolve_plan(problem);
    if (plan.schedule == W8AttnInputScheduleId::CutlassSm70) {
#ifdef NINFER_VOLTA_BUILD
        if (workspace == nullptr) {
            w8_attn_input_simt_r8_c4_launch(x, weight, q, gate, k, v, stream);
            return;
        }
        // Parent rows are stored query, key, gate, value; each output is its own allocation.
        const std::int32_t qr = q.ne[0];
        const std::int32_t kr = k.ne[0];
        const W8CutlassSm70Output outputs[4]{
            {&q, 0, qr}, {&k, qr, kr}, {&gate, qr + kr, qr}, {&v, 2 * qr + kr, kr}};
        w8_cutlass_sm70_launch(x, weight, outputs, 4, /*accumulate=*/false, *workspace, stream);
        return;
#else
        throw std::logic_error("W8 attention input: CutlassSm70 is Volta-only");
#endif
    }
    w8_attn_input_execute_plan(plan, x, weight, q, gate, k, v, stream);
}

void w8_attn_input_execute_plan(const W8AttnInputPlan& plan, const Tensor& x, const Weight& weight,
                                Tensor& q, Tensor& k, Tensor& v, cudaStream_t stream) {
    const W8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    const W8AttnInputPlan resolved = w8_attn_input_resolve_plan(problem);
    if (problem.parent_rows != 6144 || problem.kv_rows != 1024 ||
        resolved.schedule != plan.schedule) {
        throw std::invalid_argument(
            "W8 attention input: plan does not match exact three-output problem");
    }
    switch (plan.schedule) {
    case W8AttnInputScheduleId::Dflash2SimtSplit:
        w8_dflash2_attn_input_volta_launch(x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::DecodeR8Direct:
        w8_attn_input_decode_launch(x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::SplitKMmaDirect:
        w8_attn_input_splitk_mma_launch(x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::SimtR8C4:
        w8_attn_input_simt_r8_c4_launch(x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR32C128:
        w8_attn_input_mma_r32_c128_launch(x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR32C64:
        w8_companion_attn_input_mma_r32_c64_launch(x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR64C64:
        w8_companion_attn_input_mma_r64_c64_launch(x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR64C96:
        w8_companion_attn_input_mma_r64_c96_launch(x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR128C64:
        w8_companion_attn_input_mma_r128_c64_launch(x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR128C80:
        w8_companion_attn_input_mma_r128_c80_launch(x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::MmaR64C128:
        w8_attn_input_mma_r64_c128_launch(x, weight, q, k, v, stream);
        return;
    case W8AttnInputScheduleId::CutlassSm70:
        throw std::logic_error("W8 attention input: schedule not valid for this output form");
    }
    throw std::logic_error("W8 attention input: unknown schedule");
}

void w8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                            cudaStream_t stream) {
    const W8AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], weight.n, weight.padded_shape[1],
                                     x.ne[1]};
    w8_attn_input_execute_plan(w8_attn_input_resolve_plan(problem), x, weight, q, k, v, stream);
}

} // namespace ninfer::ops::detail
