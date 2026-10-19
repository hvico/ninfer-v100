// Adapted from upstream NInfer src/models/common/qwen/frontend/tool_grammar.cpp
// (github.com/Neroued/ninfer, 2026-10), reduced to the request-time tool contract of this tree.
#include "targets/qwen3_6/impl/frontend/tool_grammar.h"

#include "text/json_schema.h"

#include "grammar_builder.h"
#include "grammar_functor.h"
#include "json_schema_converter.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <utility>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {
using Json     = nlohmann::ordered_json;
using Contract = ToolGrammarContract;

std::string location(const Contract::Tool& tool, std::string_view pointer = {}) {
    return "/" + std::to_string(tool.declaration_index) + "/parameters" + std::string(pointer);
}

xgrammar::Grammar arguments(const Contract::Tool& tool) {
    try {
        std::string source;
        if (tool.strict) {
            source = text::prepare_json_schema(tool.schema_json);
        } else {
            source = R"({"type":"object","additionalProperties":{"type":"string"}})";
        }
        return xgrammar::JSONSchemaToGrammar(
            source, false, std::nullopt, std::pair<std::string, std::string>{", ", ": "}, false,
            std::nullopt, false, xgrammar::JSONFormat::kQwenXML, {});
    } catch (const RequestError& error) {
        throw RequestError(error.kind(), "tool '" + tool.name + "': " + error.what(),
                           location(tool, error.pointer()), RequestErrorSource::Tools);
    } catch (const xgrammar::JSONSchemaCompileError& error) {
        const auto kind = error.kind == xgrammar::SchemaErrorType::kUnsupportedSchema
                              ? RequestErrorKind::UnsupportedJsonSchema
                          : error.kind == xgrammar::SchemaErrorType::kUnsatisfiableSchema
                              ? RequestErrorKind::UnsatisfiableJsonSchema
                              : RequestErrorKind::InvalidJsonSchema;
        throw RequestError(kind, "tool '" + tool.name + "': " + error.what(),
                           location(tool, error.pointer), RequestErrorSource::Tools);
    } catch (const xgrammar::LogFatalError& error) {
        throw RequestError(RequestErrorKind::InvalidToolConstraint,
                           "tool '" + tool.name + "': " + error.what(), location(tool),
                           RequestErrorSource::Tools);
    }
}

xgrammar::Grammar build(const Contract& contract, const std::optional<OutputConstraint>& body) {
    std::optional<xgrammar::Grammar> json;
    if (body) { json = text::build_json_grammar(*body); }
    if (json && contract.tools.empty()) { return *json; }
    xgrammar::GrammarBuilder builder;
    std::vector<int32_t> choices;
    std::optional<int32_t> basic_arguments;
    for (const auto& tool : contract.tools) {
        const auto arguments_rule = !tool.strict && basic_arguments
                                        ? *basic_arguments
                                        : xgrammar::SubGrammarAdder::Apply(&builder, arguments(tool));
        if (!tool.strict) { basic_arguments = arguments_rule; }
        choices.push_back(builder.AddSequence({builder.AddByteString("\n<function=" + tool.name + ">\n"),
                                               builder.AddRuleRef(arguments_rule),
                                               builder.AddByteString("</function>\n</tool_call>")}));
    }
    const auto call_body = builder.AddRuleWithHint("call_body", builder.AddChoices(choices));
    const auto next_call =
        builder.AddSequence({builder.AddByteString("\n<tool_call>"), builder.AddRuleRef(call_body)});
    const auto tail  = contract.parallel ? builder.AddRepeatFromExpr("calls", next_call, 0, -1)
                                         : builder.AddEmptyStr();
    const auto first = builder.AddRuleWithHint(
        "calls", builder.AddSequence({builder.AddRuleRef(call_body), tail}));
    const auto root =
        (contract.required || body)
            ? builder.AddSequence({builder.AddByteString("<tool_call>"), builder.AddRuleRef(first)})
            : builder.AddTagDispatch({{{"<tool_call>", first}}, false, {}});
    auto calls =
        xgrammar::GrammarNormalizer::Apply(builder.Get(builder.AddRuleWithHint("root", root)));
    return json && !contract.required ? xgrammar::Grammar::Union({*json, calls}) : calls;
}

bool valid_tool_name(std::string_view name) {
    return !name.empty() && std::all_of(name.begin(), name.end(), [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
               c == '_' || c == '-' || c == '.';
    });
}
} // namespace

std::optional<ToolGrammarContract> select_tool_grammar(std::span<const std::string> tool_jsons,
                                                       const ToolChoice& choice, bool force) {
    if (tool_jsons.empty() || choice.mode == ToolChoiceMode::None) { return std::nullopt; }
    Contract contract;
    contract.required = choice.mode == ToolChoiceMode::Required;
    contract.parallel = choice.parallel;
    bool strict       = false;
    for (std::size_t index = 0; index < tool_jsons.size(); ++index) {
        const Json definition = Json::parse(tool_jsons[index], nullptr, false);
        if (definition.is_discarded() || !definition.is_object()) { continue; }
        const auto function = definition.find("function");
        if (function == definition.end() || !function->is_object()) { continue; }
        const auto name = function->find("name");
        if (name == function->end() || !name->is_string()) { continue; }
        Contract::Tool tool;
        tool.name              = name->get<std::string>();
        tool.declaration_index = index;
        if (choice.allowed_names &&
            std::find(choice.allowed_names->begin(), choice.allowed_names->end(), tool.name) ==
                choice.allowed_names->end()) {
            continue;
        }
        if (!valid_tool_name(tool.name)) {
            throw RequestError(RequestErrorKind::InvalidToolConstraint,
                               "tool name is outside the constrained output contract: " +
                                   tool.name,
                               "/" + std::to_string(index), RequestErrorSource::Tools);
        }
        if (std::any_of(contract.tools.begin(), contract.tools.end(),
                        [&](const Contract::Tool& other) { return other.name == tool.name; })) {
            throw RequestError(RequestErrorKind::InvalidToolConstraint,
                               "duplicate tool name: " + tool.name, "/" + std::to_string(index),
                               RequestErrorSource::Tools);
        }
        const auto strict_flag = function->find("strict");
        tool.strict = strict_flag != function->end() && strict_flag->is_boolean() &&
                      strict_flag->get<bool>();
        const auto parameters = function->find("parameters");
        tool.schema_json      = parameters != function->end() && parameters->is_object()
                                    ? parameters->dump()
                                    : std::string(R"({"type":"object","properties":{}})");
        strict = strict || tool.strict;
        contract.tools.push_back(std::move(tool));
    }
    if (choice.allowed_names && contract.tools.empty()) {
        throw RequestError(RequestErrorKind::InvalidToolConstraint,
                           "tool_choice selects no declared tool", {}, RequestErrorSource::Tools);
    }
    if (contract.tools.empty()) { return std::nullopt; }
    if (!force && !strict && !contract.required && !choice.allowed_names && contract.parallel) {
        return std::nullopt;
    }
    return contract;
}

std::unique_ptr<text::GrammarSession>
compile_tool_grammar(text::GrammarCompiler& compiler, const Contract& contract,
                     std::string_view reasoning_close, const std::optional<OutputConstraint>& body) {
    Json identity{{"qwen_tools", Json::array()},
                  {"required", contract.required},
                  {"parallel", contract.parallel}};
    if (body && !body->choices.empty()) {
        throw RequestError(RequestErrorKind::InvalidJsonSchema,
                           "JSON constraints do not accept literal alternatives");
    }
    if (body) {
        identity["body"] = {{"kind", static_cast<int>(body->kind)}, {"source", body->source}};
    }
    for (const auto& tool : contract.tools) {
        Json entry{{"name", tool.name}, {"strict", tool.strict}};
        if (tool.strict) {
            entry["schema"]       = Json::parse(tool.schema_json, nullptr, false);
            entry["source_index"] = tool.declaration_index;
        }
        identity["qwen_tools"].push_back(std::move(entry));
    }
    return compiler.compile_model(
        identity.dump(), [&] { return build(contract, body); }, reasoning_close, {});
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
