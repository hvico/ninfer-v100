#pragma once

#include "ninfer/types.h"
#include "text/grammar.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {

// Tool-call output language for constrained decoding. Built from the same tool JSON the chat
// template renders; only declarations selected by the request's tool choice remain.
struct ToolGrammarContract {
    struct Tool {
        std::string name;
        std::string schema_json;
        std::size_t declaration_index = 0;
        bool strict                   = false;
    };

    std::vector<Tool> tools;
    bool required = false;
    bool parallel = true;
};

// Returns a contract only when the request needs enforcement: a strict tool, a required or named
// selection, disabled parallel calls, or `force` (a body constraint alongside active tools). Free
// auto tool use keeps the unconstrained path.
[[nodiscard]] std::optional<ToolGrammarContract>
select_tool_grammar(std::span<const std::string> tool_jsons, const ToolChoice& choice,
                    bool force = false);

// Qwen's <tool_call><function=...><parameter=...> framing; strict tools constrain argument values
// by their JSON Schema. A JSON body constraint may accompany non-required tools.
[[nodiscard]] std::unique_ptr<text::GrammarSession>
compile_tool_grammar(text::GrammarCompiler& compiler, const ToolGrammarContract& contract,
                     std::string_view reasoning_close,
                     const std::optional<OutputConstraint>& body = {});

} // namespace ninfer::targets::qwen3_6::frontend_internal
