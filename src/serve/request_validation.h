#pragma once

#include "serve/request.h"
#include "serve/request_json.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::serve {

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {});

std::optional<int> optional_int(const RequestJson& object, const char* key);
std::optional<double> optional_number(const RequestJson& object, const char* key);
bool optional_bool(const RequestJson& object, const char* key, bool fallback);

[[nodiscard]] bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept;

// Output constraints (constrained decoding).
enum class JsonFormatProtocol { Chat, Responses, Anthropic };
// Chat `response_format`, Responses `text.format`, Anthropic `output_config.format`.
void parse_json_output_format(const RequestJson& format, GenerationRequest& request,
                              const std::string& param, JsonFormatProtocol protocol);
// NInfer extension `structured_outputs` ({grammar|regex|choice}) plus cross-field checks; call
// after tools, tool choice, stops and any JSON output format are parsed.
void parse_structured_outputs(const RequestJson& body, GenerationRequest& request);

} // namespace ninfer::serve
