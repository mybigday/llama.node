#pragma once

#include "chat.h"
#include "common/common.h"
#include "common/sampling.h"
#include "common/speculative.h"
#include "llama.h"
#include <cmath>
#include <cstdlib>
#include <memory>
#include <napi.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace rnllama {
ggml_type kv_cache_type_from_str(const std::string &s);
}

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

static Napi::Value json_parse(Napi::Env env, const std::string &text) {
  Napi::Object json = env.Global().Get("JSON").As<Napi::Object>();
  Napi::Function parse = json.Get("parse").As<Napi::Function>();
  return parse.Call(json, {Napi::String::New(env, text)});
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

  size_t last_end = std::string::npos;
  for (const auto &end_tag : chat_params.thinking_end_tags) {
    const size_t pos = chat_params.generation_prompt.rfind(end_tag);
    if (pos != std::string::npos &&
        (last_end == std::string::npos || pos > last_end)) {
      last_end = pos;
    }
  }
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

static void apply_speculative_draft_options(
    const Napi::Object &options, common_params_speculative_draft &draft) {
  draft.mparams.path =
      get_option<std::string>(options, "model", draft.mparams.path);
  draft.mparams.path =
      get_option<std::string>(options, "path", draft.mparams.path);
  draft.mparams.path =
      get_option<std::string>(options, "model_draft", draft.mparams.path);
  draft.mparams.path =
      get_option<std::string>(options, "draft_model", draft.mparams.path);
  draft.n_max = get_option<int32_t>(options, "n_max", draft.n_max);
  draft.n_min = get_option<int32_t>(options, "n_min", draft.n_min);
  draft.p_min = get_option<float>(options, "p_min", draft.p_min);
  draft.p_split = get_option<float>(options, "p_split", draft.p_split);
  draft.n_gpu_layers =
      get_option<int32_t>(options, "n_gpu_layers", draft.n_gpu_layers);

  const std::string cache_type_k =
      get_option<std::string>(options, "cache_type_k", "");
  if (!cache_type_k.empty()) {
    draft.cache_type_k = rnllama::kv_cache_type_from_str(cache_type_k);
  }

  const std::string cache_type_v =
      get_option<std::string>(options, "cache_type_v", "");
  if (!cache_type_v.empty()) {
    draft.cache_type_v = rnllama::kv_cache_type_from_str(cache_type_v);
  }
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

      apply_speculative_draft_options(speculative,
                                      params.speculative.draft);

      if (speculative.Has("draft") && speculative.Get("draft").IsObject()) {
        const auto draft = speculative.Get("draft").As<Napi::Object>();
        apply_speculative_draft_options(draft, params.speculative.draft);
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
  params.speculative.draft.p_split =
      get_option<float>(options, "spec_draft_p_split",
                        params.speculative.draft.p_split);
  params.speculative.draft.p_split =
      get_option<float>(options, "speculative.p_split",
                        params.speculative.draft.p_split);
  params.speculative.draft.mparams.path =
      get_option<std::string>(options, "model_draft",
                              params.speculative.draft.mparams.path);
  params.speculative.draft.mparams.path =
      get_option<std::string>(options, "draft_model",
                              params.speculative.draft.mparams.path);
  params.speculative.draft.n_gpu_layers =
      get_option<int32_t>(options, "spec_draft_n_gpu_layers",
                          params.speculative.draft.n_gpu_layers);

  const std::string draft_cache_type_k =
      get_option<std::string>(options, "spec_draft_cache_type_k", "");
  if (!draft_cache_type_k.empty()) {
    params.speculative.draft.cache_type_k =
        rnllama::kv_cache_type_from_str(draft_cache_type_k);
  }

  const std::string draft_cache_type_v =
      get_option<std::string>(options, "spec_draft_cache_type_v", "");
  if (!draft_cache_type_v.empty()) {
    params.speculative.draft.cache_type_v =
        rnllama::kv_cache_type_from_str(draft_cache_type_v);
  }

  apply_speculative_type_names(params.speculative, type_names);

  if (speculative_has_type(params.speculative,
                           COMMON_SPECULATIVE_TYPE_DRAFT_MTP) &&
      params.speculative.draft.n_max <= 0) {
    throw std::invalid_argument("MTP requires spec_draft_n_max > 0");
  }

  // A draft-simple pair with n_max <= 0 would silently run without speculation.
  if (speculative_has_type(params.speculative,
                           COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE) &&
      params.speculative.draft.n_max <= 0) {
    throw std::invalid_argument("draft-simple requires spec_draft_n_max > 0");
  }

  // Without a draft model the target itself would be used as its own draft.
  if (speculative_has_type(params.speculative,
                           COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE) &&
      !params.speculative.has_dft()) {
    throw std::invalid_argument(
        "draft-simple requires a draft model (speculative.draft.model or "
        "model_draft)");
  }
}

static bool speculative_is_draft_model(
    const common_params_speculative &speculative) {
  return speculative_has_type(speculative, COMMON_SPECULATIVE_TYPE_DRAFT_MTP) ||
         speculative_has_type(speculative, COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);
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
  const std::string thinking_end_tag =
      get_option<std::string>(options, "thinking_end_tag", "");
  std::vector<std::string> thinking_end_tags;
  if (!thinking_end_tag.empty()) {
    thinking_end_tags.push_back(thinking_end_tag);
  }

  if (chat_params != nullptr) {
    if (thinking_start_tag.empty()) {
      thinking_start_tag = chat_params->thinking_start_tag;
    }
    if (thinking_end_tags.empty()) {
      thinking_end_tags = chat_params->thinking_end_tags;
    }
  }

  if (thinking_end_tags.empty()) {
    return;
  }

  const std::string thinking_budget_message =
      get_option<std::string>(options, "thinking_budget_message", "");

  if (!thinking_start_tag.empty()) {
    sampling.reasoning_budget_start = common_tokenize(
        ctx, thinking_start_tag, /* add_special= */ false,
        /* parse_special= */ true);
  }
  for (const auto &end_tag : thinking_end_tags) {
    const auto end_tokens = common_tokenize(
        ctx, end_tag, /* add_special= */ false, /* parse_special= */ true);
    if (end_tokens.empty()) {
      continue;
    }
    if (sampling.reasoning_budget_end.empty()) {
      sampling.reasoning_budget_forced = common_tokenize(
          ctx, thinking_budget_message + end_tag,
          /* add_special= */ false, /* parse_special= */ true);
    }
    sampling.reasoning_budget_end.push_back(end_tokens);
  }

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

// A general.sampling.* value of the model's GGUF, when it has one.
static bool model_sampling_meta(const llama_model *model,
                                llama_model_meta_key key, std::string &out) {
  char buf[256] = {0};
  if (model == nullptr ||
      llama_model_meta_val_str(model, llama_model_meta_key_str(key), buf,
                               sizeof(buf)) <= 0) {
    return false;
  }
  out = buf;
  return true;
}

// A sampling option's default: the GGUF's general.sampling.* value when the
// model has one (TTS backbones carry their tuned sampling there), else
// llama.node's default.
template <typename T>
static T model_sampling_default(const llama_model *model,
                                llama_model_meta_key key, T fallback) {
  std::string v;
  if (!model_sampling_meta(model, key, v)) {
    return fallback;
  }
  char *end = nullptr;
  if constexpr (std::is_integral<T>::value) {
    const long parsed = strtol(v.c_str(), &end, 10);
    return end && end != v.c_str() ? (T)parsed : fallback;
  } else {
    const float parsed = strtof(v.c_str(), &end);
    return end && end != v.c_str() ? (T)parsed : fallback;
  }
}

// Parse the sampling options shared by completion() and queueCompletion().
// Throws std::invalid_argument on a malformed logit_bias.
// `default_samplers`: the sampler order at load (the GGUF's
// general.sampling.sequence, else llama.cpp's), used without `samplers`.
static void apply_sampling_options(
    const Napi::Object &options, const llama_model *model,
    const std::vector<common_sampler_type> &default_samplers,
    common_params_sampling &sampling) {
  auto opt_f = [&](const char *name, llama_model_meta_key key, float def) {
    return get_option<float>(options, name,
                             model_sampling_default<float>(model, key, def));
  };
  auto opt_i = [&](const char *name, llama_model_meta_key key, int32_t def) {
    return get_option<int32_t>(
        options, name, model_sampling_default<int32_t>(model, key, def));
  };

  sampling.temp = opt_f("temperature", LLAMA_MODEL_META_KEY_SAMPLING_TEMP, 0.80f);
  sampling.top_k = opt_i("top_k", LLAMA_MODEL_META_KEY_SAMPLING_TOP_K, 40);
  sampling.top_p = opt_f("top_p", LLAMA_MODEL_META_KEY_SAMPLING_TOP_P, 0.95f);
  sampling.min_p = opt_f("min_p", LLAMA_MODEL_META_KEY_SAMPLING_MIN_P, 0.05f);
  sampling.mirostat =
      opt_i("mirostat", LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT, 0);
  sampling.mirostat_tau =
      opt_f("mirostat_tau", LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU, 5.00f);
  sampling.mirostat_eta =
      opt_f("mirostat_eta", LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA, 0.10f);
  sampling.penalty_last_n =
      opt_i("penalty_last_n", LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N, 64);
  sampling.penalty_repeat = opt_f(
      "penalty_repeat", LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT, 1.00f);
  sampling.penalty_freq =
      opt_f("penalty_freq", LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_FREQ, 0.00f);
  sampling.penalty_present = opt_f(
      "penalty_present", LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_PRESENT, 0.00f);
  sampling.typ_p = get_option<float>(options, "typical_p", 1.00f);
  sampling.xtc_threshold = opt_f(
      "xtc_threshold", LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD, 0.00f);
  sampling.xtc_probability = opt_f(
      "xtc_probability", LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY, 0.10f);
  sampling.dry_multiplier = opt_f(
      "dry_multiplier", LLAMA_MODEL_META_KEY_SAMPLING_DRY_MULTIPLIER, 1.75f);
  sampling.dry_base =
      opt_f("dry_base", LLAMA_MODEL_META_KEY_SAMPLING_DRY_BASE, 2.0f);
  sampling.dry_allowed_length = opt_i(
      "dry_allowed_length", LLAMA_MODEL_META_KEY_SAMPLING_DRY_ALLOWED_LENGTH,
      -1);
  // 0 (or a negative value) disables DRY, as in llama.cpp since #26524.
  sampling.dry_penalty_last_n = opt_i(
      "dry_penalty_last_n", LLAMA_MODEL_META_KEY_SAMPLING_DRY_PENALTY_LAST_N,
      0);
  sampling.top_n_sigma = get_option<float>(options, "top_n_sigma", -1.0f);
  sampling.ignore_eos = get_option<bool>(options, "ignore_eos", false);
  sampling.seed = get_option<int32_t>(options, "seed", LLAMA_DEFAULT_SEED);
  sampling.n_probs = get_option<int32_t>(options, "n_probs", 0);

  if (options.Has("dry_sequence_breakers") &&
      options.Get("dry_sequence_breakers").IsArray()) {
    auto breakers = options.Get("dry_sequence_breakers").As<Napi::Array>();
    sampling.dry_sequence_breakers.clear();
    for (size_t i = 0; i < breakers.Length(); i++) {
      sampling.dry_sequence_breakers.push_back(
          breakers.Get(i).ToString().Utf8Value());
    }
  }

  // Sampler order: "penalties;top_k;..." or an array of names.
  sampling.samplers = default_samplers;
  {
    std::vector<std::string> names;
    auto value = options.Has("samplers") ? options.Get("samplers")
                                         : options.Env().Undefined();
    if (value.IsString()) {
      const std::string s = value.ToString().Utf8Value();
      size_t start = 0;
      while (start <= s.size()) {
        size_t end = s.find(';', start);
        if (end == std::string::npos) end = s.size();
        if (end > start) names.push_back(s.substr(start, end - start));
        start = end + 1;
      }
    } else if (value.IsArray()) {
      auto arr = value.As<Napi::Array>();
      for (size_t i = 0; i < arr.Length(); i++) {
        if (arr.Get(i).IsString()) {
          names.push_back(arr.Get(i).ToString().Utf8Value());
        }
      }
    }
    if (!names.empty()) {
      sampling.samplers = common_sampler_types_from_names(names);
    }
  }

  // logit_bias: [[token, bias | false], ...], or a Float32Array of one bias
  // per vocabulary token (0: none, -Infinity: never sampled).
  sampling.logit_bias.clear();
  if (options.Has("logit_bias")) {
    auto value = options.Get("logit_bias");
    if (value.IsTypedArray() &&
        value.As<Napi::TypedArray>().TypedArrayType() == napi_float32_array) {
      auto table = value.As<Napi::Float32Array>();
      const int n_vocab =
          model ? llama_vocab_n_tokens(llama_model_get_vocab(model)) : 0;
      if ((int)table.ElementLength() != n_vocab) {
        throw std::invalid_argument(
            "logit_bias: a Float32Array needs one entry per vocabulary token (" +
            std::to_string(n_vocab) + "), got " +
            std::to_string(table.ElementLength()));
      }
      const float *data = table.Data();
      for (int t = 0; t < n_vocab; ++t) {
        if (data[t] != 0.0f) sampling.logit_bias.push_back({t, data[t]});
      }
    } else if (value.IsArray()) {
      auto arr = value.As<Napi::Array>();
      for (size_t i = 0; i < arr.Length(); i++) {
        auto item = arr.Get(i);
        if (!item.IsArray()) continue;
        auto pair = item.As<Napi::Array>();
        if (pair.Length() != 2 || !pair.Get(0u).IsNumber()) continue;
        const llama_token token = pair.Get(0u).ToNumber().Int32Value();
        auto bias = pair.Get(1u);
        if (bias.IsNumber()) {
          sampling.logit_bias.push_back({token, bias.ToNumber().FloatValue()});
        } else if (bias.IsBoolean() && !bias.ToBoolean().Value()) {
          sampling.logit_bias.push_back({token, -INFINITY});
        }
      }
    }
  }
}
