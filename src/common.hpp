#pragma once

#include "chat.h"
#include "common/common.h"
#include "common/sampling.h"
#include "common/speculative.h"
#include "llama.h"
#include <memory>
#include <napi.h>
#include <stdexcept>
#include <string>
#include <vector>

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

static std::string normalize_speculative_type_name(std::string name) {
  if (name == "mtp") {
    return "draft-mtp";
  }
  return name;
}

static bool speculative_has_type(const common_params_speculative &speculative,
                                 common_speculative_type type) {
  return std::find(speculative.types.begin(), speculative.types.end(), type) !=
         speculative.types.end();
}

static void apply_speculative_type_names(
    common_params_speculative &speculative,
    std::vector<std::string> type_names) {
  if (type_names.empty()) {
    return;
  }

  for (auto &name : type_names) {
    name = normalize_speculative_type_name(name);
  }

  speculative.types = common_speculative_types_from_names(type_names);
}

static void apply_speculative_options(const Napi::Object &options,
                                      common_params &params) {
  std::vector<std::string> type_names;

  if (options.Has("spec_type") && !is_nil(options.Get("spec_type"))) {
    const auto value = options.Get("spec_type");
    if (value.IsArray()) {
      const auto array = value.As<Napi::Array>();
      for (uint32_t i = 0; i < array.Length(); i++) {
        type_names.push_back(array.Get(i).ToString().Utf8Value());
      }
    } else {
      type_names.push_back(value.ToString().Utf8Value());
    }
  }

  if (options.Has("speculative") && !is_nil(options.Get("speculative"))) {
    const auto value = options.Get("speculative");
    if (value.IsBoolean()) {
      type_names.push_back(value.ToBoolean().Value() ? "draft-mtp" : "none");
    } else if (value.IsString()) {
      type_names.push_back(value.ToString().Utf8Value());
    } else if (value.IsObject()) {
      const auto speculative = value.As<Napi::Object>();

      if (speculative.Has("enabled") &&
          !speculative.Get("enabled").ToBoolean().Value()) {
        type_names.push_back("none");
      }

      if (speculative.Has("type") && !is_nil(speculative.Get("type"))) {
        type_names.push_back(speculative.Get("type").ToString().Utf8Value());
      }

      if (speculative.Has("types") && speculative.Get("types").IsArray()) {
        const auto types = speculative.Get("types").As<Napi::Array>();
        for (uint32_t i = 0; i < types.Length(); i++) {
          type_names.push_back(types.Get(i).ToString().Utf8Value());
        }
      }

      params.speculative.draft.n_max =
          get_option<int32_t>(speculative, "n_max",
                              params.speculative.draft.n_max);
      params.speculative.draft.n_min =
          get_option<int32_t>(speculative, "n_min",
                              params.speculative.draft.n_min);
      params.speculative.draft.p_min =
          get_option<float>(speculative, "p_min",
                            params.speculative.draft.p_min);
      params.speculative.draft.p_split =
          get_option<float>(speculative, "p_split",
                            params.speculative.draft.p_split);

      if (speculative.Has("draft") && speculative.Get("draft").IsObject()) {
        const auto draft = speculative.Get("draft").As<Napi::Object>();
        params.speculative.draft.n_max =
            get_option<int32_t>(draft, "n_max",
                                params.speculative.draft.n_max);
        params.speculative.draft.n_min =
            get_option<int32_t>(draft, "n_min",
                                params.speculative.draft.n_min);
        params.speculative.draft.p_min =
            get_option<float>(draft, "p_min",
                              params.speculative.draft.p_min);
        params.speculative.draft.p_split =
            get_option<float>(draft, "p_split",
                              params.speculative.draft.p_split);
      }
    }
  }

  params.speculative.draft.n_max =
      get_option<int32_t>(options, "spec_draft_n_max",
                          params.speculative.draft.n_max);
  params.speculative.draft.n_max =
      get_option<int32_t>(options, "speculative.n_max",
                          params.speculative.draft.n_max);
  params.speculative.draft.n_min =
      get_option<int32_t>(options, "spec_draft_n_min",
                          params.speculative.draft.n_min);
  params.speculative.draft.n_min =
      get_option<int32_t>(options, "speculative.n_min",
                          params.speculative.draft.n_min);
  params.speculative.draft.p_min =
      get_option<float>(options, "spec_draft_p_min",
                        params.speculative.draft.p_min);
  params.speculative.draft.p_min =
      get_option<float>(options, "speculative.p_min",
                        params.speculative.draft.p_min);

  apply_speculative_type_names(params.speculative, type_names);

  if (speculative_has_type(params.speculative,
                           COMMON_SPECULATIVE_TYPE_DRAFT_MTP) &&
      params.speculative.draft.n_max <= 0) {
    throw std::invalid_argument("MTP requires spec_draft_n_max > 0");
  }
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
