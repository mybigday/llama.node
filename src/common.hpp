#pragma once

#include "chat.h"
#include "common/common.h"
#include "common/sampling.h"
#include "llama.h"
#include <memory>
#include <napi.h>
#include <string>

typedef std::unique_ptr<common_sampler, decltype(&common_sampler_free)>
    LlamaCppSampling;
typedef std::unique_ptr<llama_batch, decltype(&llama_batch_free)> LlamaCppBatch;

static bool is_nil(const Napi::Value &value) {
  return value.IsNull() || value.IsUndefined();
}

// Overload for Napi::Value to handle both arrays and objects
static std::string json_stringify(const Napi::Value &value) {
  Napi::Env env = value.Env();
  Napi::Object json = env.Global().Get("JSON").As<Napi::Object>();
  Napi::Function stringify = json.Get("stringify").As<Napi::Function>();
  return stringify.Call(json, {value}).As<Napi::String>().ToString();
}

static void console_log(Napi::Env env, const std::string &message) {
  Napi::Function consoleLog = env.Global()
                                  .Get("console")
                                  .As<Napi::Object>()
                                  .Get("log")
                                  .As<Napi::Function>();
  consoleLog.Call({Napi::String::New(env, message)});
}

template <typename T>
constexpr T get_option(const Napi::Object &options, const std::string &name,
                       const T default_value) {
  if (options.Has(name) && !is_nil(options.Get(name))) {
    if constexpr (std::is_same<T, std::string>::value) {
      return options.Get(name).ToString().operator T();
    } else if constexpr (std::is_same<T, int32_t>::value ||
                         std::is_same<T, uint32_t>::value ||
                         std::is_same<T, float>::value ||
                         std::is_same<T, double>::value) {
      return options.Get(name).ToNumber().operator T();
    } else if constexpr (std::is_same<T, bool>::value) {
      return options.Get(name).ToBoolean().operator T();
    } else {
      static_assert(std::is_same<T, std::string>::value ||
                        std::is_same<T, int32_t>::value ||
                        std::is_same<T, uint32_t>::value ||
                        std::is_same<T, float>::value ||
                        std::is_same<T, double>::value ||
                        std::is_same<T, bool>::value,
                    "Unsupported type");
    }
  } else {
    return default_value;
  }
}

static bool is_thinking_forced_open(
    const common_chat_params &chat_params) {
  if (!chat_params.supports_thinking ||
      chat_params.thinking_start_tag.empty()) {
    return false;
  }

  const size_t last_start =
      chat_params.generation_prompt.rfind(chat_params.thinking_start_tag);
  if (last_start == std::string::npos) {
    return false;
  }

  if (chat_params.thinking_end_tag.empty()) {
    return true;
  }

  const size_t last_end =
      chat_params.generation_prompt.rfind(chat_params.thinking_end_tag);
  return last_end == std::string::npos || last_end < last_start;
}

static void reset_reasoning_budget(common_params_sampling &sampling) {
  sampling.reasoning_budget_tokens = -1;
  sampling.reasoning_budget_activate_immediately = false;
  sampling.reasoning_budget_start.clear();
  sampling.reasoning_budget_end.clear();
  sampling.reasoning_budget_forced.clear();
}

static void apply_reasoning_budget(
    const Napi::Object &options, llama_context *ctx,
    common_params_sampling &sampling,
    const common_chat_params *chat_params = nullptr) {
  reset_reasoning_budget(sampling);

  const int32_t thinking_budget_tokens =
      get_option<int32_t>(options, "thinking_budget_tokens", -1);
  if (thinking_budget_tokens < 0) {
    return;
  }

  std::string thinking_start_tag =
      get_option<std::string>(options, "thinking_start_tag", "");
  std::string thinking_end_tag =
      get_option<std::string>(options, "thinking_end_tag", "");

  if (chat_params != nullptr) {
    if (thinking_start_tag.empty()) {
      thinking_start_tag = chat_params->thinking_start_tag;
    }
    if (thinking_end_tag.empty()) {
      thinking_end_tag = chat_params->thinking_end_tag;
    }
  }

  if (thinking_end_tag.empty()) {
    return;
  }

  const std::string thinking_budget_message =
      get_option<std::string>(options, "thinking_budget_message", "");

  if (!thinking_start_tag.empty()) {
    sampling.reasoning_budget_start = common_tokenize(
        ctx, thinking_start_tag, /* add_special= */ false,
        /* parse_special= */ true);
  }
  sampling.reasoning_budget_end = common_tokenize(
      ctx, thinking_end_tag, /* add_special= */ false,
      /* parse_special= */ true);
  sampling.reasoning_budget_forced = common_tokenize(
      ctx, thinking_budget_message + thinking_end_tag,
      /* add_special= */ false, /* parse_special= */ true);

  if (sampling.reasoning_budget_end.empty() ||
      sampling.reasoning_budget_forced.empty()) {
    reset_reasoning_budget(sampling);
    return;
  }

  sampling.reasoning_budget_tokens = thinking_budget_tokens;

  bool thinking_forced_open =
      get_option<bool>(options, "thinking_forced_open", false);
  if (!thinking_forced_open && chat_params != nullptr) {
    thinking_forced_open = is_thinking_forced_open(*chat_params);
  }
  sampling.reasoning_budget_activate_immediately = thinking_forced_open;
}
