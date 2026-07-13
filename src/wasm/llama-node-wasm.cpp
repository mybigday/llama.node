#include "rn-llama/rn-completion.h"
#include "rn-llama/rn-llama.h"
#include "json-schema-to-grammar.h"
#include "common/speculative.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

using json = nlohmann::ordered_json;

namespace {

std::unique_ptr<rnllama::llama_rn_context> g_ctx;
std::string g_result;
bool g_started = false;

template <typename T>
T opt(const json &obj, const char *name, T fallback) {
  if (!obj.is_object() || !obj.contains(name) || obj.at(name).is_null()) {
    return fallback;
  }
  return obj.at(name).get<T>();
}

std::string opt_string(const json &obj, const char *name,
                       const std::string &fallback = "") {
  if (!obj.is_object() || !obj.contains(name) || obj.at(name).is_null()) {
    return fallback;
  }
  if (obj.at(name).is_string()) {
    return obj.at(name).get<std::string>();
  }
  return obj.at(name).dump();
}

json ok(json data = json::object()) {
  data["success"] = true;
  return data;
}

json error_json(const std::string &message) {
  return {{"success", false}, {"error", message}};
}

const char *set_result(const json &value) {
  g_result = value.dump(-1, ' ', false, json::error_handler_t::replace);
  return g_result.c_str();
}

void require_context() {
  if (!g_ctx || !g_ctx->ctx || !g_ctx->model) {
    throw std::runtime_error("Model not loaded");
  }
}

std::vector<std::string> string_array(const json &obj, const char *name) {
  std::vector<std::string> values;
  if (!obj.is_object() || !obj.contains(name) || !obj.at(name).is_array()) {
    return values;
  }
  for (const auto &item : obj.at(name)) {
    values.push_back(item.get<std::string>());
  }
  return values;
}

std::string normalize_speculative_type_name(std::string name) {
  if (name == "mtp") {
    return "draft-mtp";
  }
  return name;
}

bool speculative_has_type(const common_params_speculative &speculative,
                          common_speculative_type type) {
  return std::find(speculative.types.begin(), speculative.types.end(), type) !=
         speculative.types.end();
}

void push_speculative_type_names(const json &value,
                                 std::vector<std::string> &type_names) {
  if (value.is_array()) {
    for (const auto &item : value) {
      type_names.push_back(item.is_string() ? item.get<std::string>()
                                            : item.dump());
    }
    return;
  }
  type_names.push_back(value.is_string() ? value.get<std::string>()
                                         : value.dump());
}

void apply_speculative_type_names(common_params_speculative &speculative,
                                  std::vector<std::string> type_names) {
  if (type_names.empty()) {
    return;
  }

  for (auto &name : type_names) {
    name = normalize_speculative_type_name(name);
  }

  speculative.types = common_speculative_types_from_names(type_names);
}

void apply_speculative_draft_options_json(
    const json &options, common_params_speculative_draft &draft) {
  draft.mparams.path = opt_string(options, "model", draft.mparams.path);
  draft.mparams.path = opt_string(options, "path", draft.mparams.path);
  draft.mparams.path = opt_string(options, "model_draft", draft.mparams.path);
  draft.mparams.path = opt_string(options, "draft_model", draft.mparams.path);
  draft.n_max = opt<int32_t>(options, "n_max", draft.n_max);
  draft.n_min = opt<int32_t>(options, "n_min", draft.n_min);
  draft.p_min = opt<float>(options, "p_min", draft.p_min);
  draft.p_split = opt<float>(options, "p_split", draft.p_split);
  draft.n_gpu_layers = opt<int32_t>(options, "n_gpu_layers", draft.n_gpu_layers);

  const std::string cache_type_k = opt_string(options, "cache_type_k");
  if (!cache_type_k.empty()) {
    draft.cache_type_k = rnllama::kv_cache_type_from_str(cache_type_k);
  }

  const std::string cache_type_v = opt_string(options, "cache_type_v");
  if (!cache_type_v.empty()) {
    draft.cache_type_v = rnllama::kv_cache_type_from_str(cache_type_v);
  }
}

void apply_speculative_options_json(const json &options,
                                    common_params &params) {
  std::vector<std::string> type_names;

  if (options.contains("spec_type") && !options.at("spec_type").is_null()) {
    push_speculative_type_names(options.at("spec_type"), type_names);
  }

  if (options.contains("speculative") && !options.at("speculative").is_null()) {
    const auto &value = options.at("speculative");
    if (value.is_boolean()) {
      type_names.push_back(value.get<bool>() ? "draft-mtp" : "none");
    } else if (value.is_string()) {
      type_names.push_back(value.get<std::string>());
    } else if (value.is_object()) {
      if (value.contains("enabled") && value.at("enabled").is_boolean() &&
          !value.at("enabled").get<bool>()) {
        type_names.push_back("none");
      }

      if (value.contains("type") && !value.at("type").is_null()) {
        type_names.push_back(opt_string(value, "type"));
      }

      if (value.contains("types") && value.at("types").is_array()) {
        push_speculative_type_names(value.at("types"), type_names);
      }

      apply_speculative_draft_options_json(value, params.speculative.draft);

      if (value.contains("draft") && value.at("draft").is_object()) {
        const auto &draft = value.at("draft");
        apply_speculative_draft_options_json(draft,
                                             params.speculative.draft);
      }
    }
  }

  params.speculative.draft.n_max =
      opt<int32_t>(options, "spec_draft_n_max",
                   params.speculative.draft.n_max);
  params.speculative.draft.n_max =
      opt<int32_t>(options, "speculative.n_max",
                   params.speculative.draft.n_max);
  params.speculative.draft.n_min =
      opt<int32_t>(options, "spec_draft_n_min",
                   params.speculative.draft.n_min);
  params.speculative.draft.n_min =
      opt<int32_t>(options, "speculative.n_min",
                   params.speculative.draft.n_min);
  params.speculative.draft.p_min =
      opt<float>(options, "spec_draft_p_min", params.speculative.draft.p_min);
  params.speculative.draft.p_min =
      opt<float>(options, "speculative.p_min", params.speculative.draft.p_min);
  params.speculative.draft.p_split =
      opt<float>(options, "spec_draft_p_split",
                 params.speculative.draft.p_split);
  params.speculative.draft.p_split =
      opt<float>(options, "speculative.p_split",
                 params.speculative.draft.p_split);
  params.speculative.draft.mparams.path =
      opt_string(options, "model_draft",
                 params.speculative.draft.mparams.path);
  params.speculative.draft.mparams.path =
      opt_string(options, "draft_model",
                 params.speculative.draft.mparams.path);
  params.speculative.draft.n_gpu_layers =
      opt<int32_t>(options, "spec_draft_n_gpu_layers",
                   params.speculative.draft.n_gpu_layers);

  const std::string draft_cache_type_k =
      opt_string(options, "spec_draft_cache_type_k");
  if (!draft_cache_type_k.empty()) {
    params.speculative.draft.cache_type_k =
        rnllama::kv_cache_type_from_str(draft_cache_type_k);
  }

  const std::string draft_cache_type_v =
      opt_string(options, "spec_draft_cache_type_v");
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
}

bool is_thinking_forced_open(const common_chat_params &chat_params) {
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

void reset_reasoning_budget(common_params_sampling &sampling) {
  sampling.reasoning_budget_tokens = -1;
  sampling.reasoning_budget_activate_immediately = false;
  sampling.reasoning_budget_start.clear();
  sampling.reasoning_budget_end.clear();
  sampling.reasoning_budget_forced.clear();
  sampling.reasoning_budget_message.clear();
}

void apply_reasoning_budget_json(const json &options, llama_context *ctx,
                                 common_params_sampling &sampling,
                                 const common_chat_params *chat_params) {
  reset_reasoning_budget(sampling);

  const int32_t thinking_budget_tokens =
      opt<int32_t>(options, "thinking_budget_tokens", -1);
  if (thinking_budget_tokens < 0) {
    return;
  }

  std::string thinking_start_tag = opt_string(options, "thinking_start_tag");
  std::string thinking_end_tag = opt_string(options, "thinking_end_tag");

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
      opt_string(options, "thinking_budget_message");

  if (!thinking_start_tag.empty()) {
    sampling.reasoning_budget_start =
        common_tokenize(ctx, thinking_start_tag, false, true);
  }
  sampling.reasoning_budget_end =
      common_tokenize(ctx, thinking_end_tag, false, true);
  sampling.reasoning_budget_forced = common_tokenize(
      ctx, thinking_budget_message + thinking_end_tag, false, true);

  if (sampling.reasoning_budget_end.empty() ||
      sampling.reasoning_budget_forced.empty()) {
    reset_reasoning_budget(sampling);
    return;
  }

  sampling.reasoning_budget_tokens = thinking_budget_tokens;
  sampling.reasoning_budget_message = thinking_budget_message;

  bool thinking_forced_open =
      opt<bool>(options, "thinking_forced_open", false);
  if (!thinking_forced_open && chat_params != nullptr) {
    thinking_forced_open = is_thinking_forced_open(*chat_params);
  }
  sampling.reasoning_budget_activate_immediately = thinking_forced_open;
}

void apply_grammar_trigger_options(common_params &params, const json &options) {
  if (!options.contains("grammar_triggers") ||
      !options.at("grammar_triggers").is_array()) {
    return;
  }

  for (const auto &item : options.at("grammar_triggers")) {
    if (!item.is_object()) {
      throw std::runtime_error("grammar_triggers must contain objects");
    }

    const auto type = static_cast<common_grammar_trigger_type>(
        opt<int32_t>(item, "type", COMMON_GRAMMAR_TRIGGER_TYPE_WORD));
    const std::string word = opt_string(item, "value");

    if (type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
      const auto ids = common_tokenize(g_ctx->ctx, word, false, true);
      if (ids.size() == 1) {
        const llama_token token = ids[0];
        if (params.sampling.preserved_tokens.find(token) ==
            params.sampling.preserved_tokens.end()) {
          throw std::runtime_error(
              "Grammar trigger word should be marked as preserved token");
        }

        common_grammar_trigger trigger;
        trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
        trigger.value = word;
        trigger.token = token;
        params.sampling.grammar_triggers.push_back(std::move(trigger));
      } else {
        params.sampling.grammar_triggers.push_back(
            {COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
      }
      continue;
    }

    common_grammar_trigger trigger;
    trigger.type = type;
    trigger.value = word;
    if (type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
      trigger.token =
          static_cast<llama_token>(opt<int32_t>(item, "token", LLAMA_TOKEN_NULL));
    }
    params.sampling.grammar_triggers.push_back(std::move(trigger));
  }
}

void append_chat_template_grammar_trigger(
    llama_context *ctx, common_params_sampling &sampling,
    const common_grammar_trigger &trigger) {
  if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
    const auto ids = common_tokenize(ctx, trigger.value, false, true);
    if (ids.size() == 1 &&
        sampling.preserved_tokens.find(ids[0]) !=
            sampling.preserved_tokens.end()) {
      common_grammar_trigger token_trigger;
      token_trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
      token_trigger.value = trigger.value;
      token_trigger.token = ids[0];
      sampling.grammar_triggers.push_back(std::move(token_trigger));
      return;
    }
  }

  sampling.grammar_triggers.push_back(trigger);
}

json token_probs_json(
    llama_context *ctx,
    const std::vector<rnllama::completion_token_output> &probs) {
  json result = json::array();
  for (const auto &prob : probs) {
    json item = {
        {"content", rnllama::tokens_to_output_formatted_string(ctx, prob.tok)},
        {"probs", json::array()},
    };
    for (const auto &p : prob.probs) {
      item["probs"].push_back(
          {{"tok_str", rnllama::tokens_to_output_formatted_string(ctx, p.tok)},
           {"prob", p.prob}});
    }
    result.push_back(item);
  }
  return result;
}

int32_t pooling_type_from_str(const std::string &s) {
  if (s == "none")
    return LLAMA_POOLING_TYPE_NONE;
  if (s == "mean")
    return LLAMA_POOLING_TYPE_MEAN;
  if (s == "cls")
    return LLAMA_POOLING_TYPE_CLS;
  if (s == "last")
    return LLAMA_POOLING_TYPE_LAST;
  if (s == "rank")
    return LLAMA_POOLING_TYPE_RANK;
  return LLAMA_POOLING_TYPE_UNSPECIFIED;
}

void backend_init(bool include_gpu) {
  static bool base_initialized = false;

  if (!base_initialized) {
    ggml_time_init();
    struct ggml_init_params params = {0, nullptr, false};
    struct ggml_context *ctx = ggml_init(params);
    ggml_free(ctx);
    base_initialized = true;
  }

  if (include_gpu) {
    ggml_backend_load_all();
    return;
  }

  if (ggml_backend_reg_by_name("CPU") == nullptr) {
    ggml_backend_register(ggml_backend_cpu_reg());
  }
}

json model_info_json() {
  require_context();

  char desc[1024];
  llama_model_desc(g_ctx->model, desc, sizeof(desc));

  json metadata = json::object();
  const int count = llama_model_meta_count(g_ctx->model);
  for (int i = 0; i < count; i++) {
    char key[256];
    char val[16384];
    llama_model_meta_key_by_index(g_ctx->model, i, key, sizeof(key));
    llama_model_meta_val_str_by_index(g_ctx->model, i, val, sizeof(val));
    metadata[key] = val;
  }

  json jinja = json::object();
  jinja["default"] = g_ctx->validateModelChatTemplate(true, nullptr);
  const auto default_caps =
      common_chat_templates_get_caps_for_variant(g_ctx->templates.get(), "");
  jinja["defaultCaps"] = {
      {"tools", default_caps.supports_tools},
      {"toolCalls", default_caps.supports_tool_calls},
      {"systemRole", default_caps.supports_system_role},
      {"parallelToolCalls", default_caps.supports_parallel_tool_calls},
  };

  const bool has_tool_use =
      common_chat_templates_has_variant(g_ctx->templates.get(), "tool_use");
  jinja["toolUse"] = has_tool_use;
  if (has_tool_use) {
    const auto caps =
        common_chat_templates_get_caps_for_variant(g_ctx->templates.get(),
                                                   "tool_use");
    jinja["toolUseCaps"] = {
        {"tools", caps.supports_tools},
        {"toolCalls", caps.supports_tool_calls},
        {"systemRole", caps.supports_system_role},
        {"parallelToolCalls", caps.supports_parallel_tool_calls},
    };
  }

  json chat_templates = {
      {"llamaChat", g_ctx->validateModelChatTemplate(false, nullptr)},
      {"jinja", jinja},
  };

  return {
      {"desc", desc},
      {"nEmbd", llama_model_n_embd(g_ctx->model)},
      {"nParams", llama_model_n_params(g_ctx->model)},
      {"size", llama_model_size(g_ctx->model)},
      {"is_recurrent", llama_model_is_recurrent(g_ctx->model)},
      {"is_hybrid", llama_model_is_hybrid(g_ctx->model)},
      {"chatTemplates", chat_templates},
      {"isChatTemplateSupported",
       g_ctx->validateModelChatTemplate(false, nullptr)},
      {"metadata", metadata},
  };
}

common_params params_from_load_options(const json &options) {
  common_params params;
  params.fit_params = false;
  params.model.path = opt_string(options, "model");
  if (params.model.path.empty()) {
    throw std::runtime_error("Model is required");
  }

  params.vocab_only = opt<bool>(options, "vocab_only", false);
  if (params.vocab_only) {
    params.warmup = false;
  }

  params.chat_template = opt_string(options, "chat_template");
  params.n_ctx = opt<int32_t>(options, "n_ctx", 512);
  params.n_batch = opt<int32_t>(options, "n_batch", 2048);
  params.n_ubatch = opt<int32_t>(options, "n_ubatch", 512);
  params.n_parallel = opt<int32_t>(options, "n_parallel", 1);
  params.embedding = opt<bool>(options, "embedding", false);
  if (params.embedding) {
    params.n_ubatch = params.n_batch;
  }
  params.embd_normalize = opt<int32_t>(options, "embd_normalize", 2);
  params.pooling_type = static_cast<enum llama_pooling_type>(
      pooling_type_from_str(opt_string(options, "pooling_type")));
  params.cpuparams.n_threads = opt<int32_t>(options, "n_threads", 1);
  params.n_gpu_layers = opt<int32_t>(options, "n_gpu_layers", 0);
  params.flash_attn_type =
      rnllama::flash_attn_type_from_str(opt_string(options, "flash_attn_type",
                                                   "auto"));
  if (options.contains("flash_attn")) {
    params.flash_attn_type =
        opt<bool>(options, "flash_attn", false)
            ? LLAMA_FLASH_ATTN_TYPE_ENABLED
            : LLAMA_FLASH_ATTN_TYPE_DISABLED;
  }
  params.cache_type_k =
      rnllama::kv_cache_type_from_str(opt_string(options, "cache_type_k", "f16"));
  params.cache_type_v =
      rnllama::kv_cache_type_from_str(opt_string(options, "cache_type_v", "f16"));
  params.ctx_shift = opt<bool>(options, "ctx_shift", true);
  params.kv_unified = opt<bool>(options, "kv_unified", false);
  params.swa_full = opt<bool>(options, "swa_full", false);
  params.rope_freq_base = opt<float>(options, "rope_freq_base", 0.0f);
  params.rope_freq_scale = opt<float>(options, "rope_freq_scale", 0.0f);
  params.use_mlock = false;
  params.use_mmap = opt<bool>(options, "use_mmap", true);
  params.no_extra_bufts = opt<bool>(options, "no_extra_bufts", false);
  apply_speculative_options_json(options, params);

  return params;
}

json action_load(const json &payload) {
  common_params params = params_from_load_options(payload);
  backend_init(params.n_gpu_layers > 0);

  g_ctx.reset(new rnllama::llama_rn_context());

  if (!g_ctx->loadModel(params)) {
    g_ctx.reset();
    throw std::runtime_error("Failed to load model");
  }
  g_ctx->attachThreadpoolsIfAvailable();

  return ok({{"modelInfo", model_info_json()},
             {"systemInfo", common_params_get_system_info(params)}});
}

std::string response_schema_string(const json &options) {
  if (!options.contains("response_format") ||
      !options.at("response_format").is_object()) {
    return "";
  }
  const auto &format = options.at("response_format");
  const std::string type = opt_string(format, "type", "text");
  if (type == "json_schema" && format.contains("json_schema") &&
      format.at("json_schema").is_object()) {
    const auto &schema_parent = format.at("json_schema");
    if (schema_parent.contains("schema")) {
      return schema_parent.at("schema").dump();
    }
    return "{}";
  }
  if (type == "json_object") {
    if (format.contains("schema")) {
      return format.at("schema").dump();
    }
    return "{}";
  }
  return "";
}

common_chat_params format_jinja_chat(
    const json &messages, const json &options, const std::string &json_schema,
    const std::string &reasoning_format) {
  std::map<std::string, std::string> kwargs;
  if (options.contains("chat_template_kwargs") &&
      options.at("chat_template_kwargs").is_object()) {
    for (const auto &[key, value] : options.at("chat_template_kwargs").items()) {
      kwargs[key] = value.is_string() ? value.get<std::string>() : value.dump();
    }
  }

  const std::string tools_str =
      options.contains("tools") && options.at("tools").is_array()
          ? options.at("tools").dump()
          : "";

  return g_ctx->getFormattedChatWithJinja(
      messages.dump(), opt_string(options, "chat_template"), json_schema,
      tools_str, opt<bool>(options, "parallel_tool_calls", false),
      opt_string(options, "tool_choice", "none"),
      opt<bool>(options, "enable_thinking", true), reasoning_format,
      opt<bool>(options, "add_generation_prompt", true),
      opt_string(options, "now"), kwargs,
      opt<bool>(options, "force_pure_content", false));
}

void apply_preserved_tokens(common_params &params, const json &options) {
  auto add_token = [&](const std::string &token) {
    const auto ids = common_tokenize(g_ctx->ctx, token, false, true);
    if (ids.size() == 1) {
      params.sampling.preserved_tokens.insert(ids[0]);
    }
  };

  if (options.contains("preserved_tokens") &&
      options.at("preserved_tokens").is_array()) {
    for (const auto &token : options.at("preserved_tokens")) {
      add_token(token.get<std::string>());
    }
  }
}

void apply_sampling_options(common_params &params, const json &options) {
  params.n_predict = opt<int32_t>(
      options, "n_predict",
      opt<int32_t>(options, "max_tokens", opt<int32_t>(options, "max_length", -1)));
  params.sampling.temp = opt<float>(options, "temperature", 0.80f);
  params.sampling.top_k = opt<int32_t>(options, "top_k", 40);
  params.sampling.top_p = opt<float>(options, "top_p", 0.95f);
  params.sampling.min_p = opt<float>(options, "min_p", 0.05f);
  params.sampling.mirostat = opt<int32_t>(options, "mirostat", 0);
  params.sampling.mirostat_tau = opt<float>(options, "mirostat_tau", 5.0f);
  params.sampling.mirostat_eta = opt<float>(options, "mirostat_eta", 0.1f);
  params.sampling.penalty_last_n =
      opt<int32_t>(options, "penalty_last_n", 64);
  params.sampling.penalty_repeat =
      opt<float>(options, "penalty_repeat", 1.0f);
  params.sampling.penalty_freq = opt<float>(options, "penalty_freq", 0.0f);
  params.sampling.penalty_present =
      opt<float>(options, "penalty_present", 0.0f);
  params.sampling.typ_p =
      opt<float>(options, "typ_p", opt<float>(options, "typical_p", 1.0f));
  params.sampling.xtc_threshold =
      opt<float>(options, "xtc_threshold", 0.0f);
  params.sampling.xtc_probability =
      opt<float>(options, "xtc_probability", 0.1f);
  params.sampling.dry_multiplier =
      opt<float>(options, "dry_multiplier", 1.75f);
  params.sampling.dry_base = opt<float>(options, "dry_base", 2.0f);
  params.sampling.dry_allowed_length =
      opt<float>(options, "dry_allowed_length", -1.0f);
  params.sampling.dry_penalty_last_n =
      opt<float>(options, "dry_penalty_last_n", 0.0f);
  params.sampling.top_n_sigma = opt<float>(options, "top_n_sigma", -1.0f);
  params.sampling.ignore_eos = opt<bool>(options, "ignore_eos", false);
  params.n_keep = opt<int32_t>(options, "n_keep", 0);
  params.sampling.seed = opt<int32_t>(options, "seed", LLAMA_DEFAULT_SEED);
  params.sampling.n_probs = opt<int32_t>(options, "n_probs", 0);

  if (options.contains("dry_sequence_breakers") &&
      options.at("dry_sequence_breakers").is_array()) {
    params.sampling.dry_sequence_breakers.clear();
    for (const auto &breaker : options.at("dry_sequence_breakers")) {
      params.sampling.dry_sequence_breakers.push_back(
          breaker.is_string() ? breaker.get<std::string>() : breaker.dump());
    }
  }

  if (options.contains("logit_bias") && options.at("logit_bias").is_array()) {
    params.sampling.logit_bias.clear();
    for (const auto &item : options.at("logit_bias")) {
      if (item.is_array() && item.size() == 2) {
        params.sampling.logit_bias.push_back(
            {static_cast<llama_token>(item.at(0).get<int32_t>()),
             item.at(1).get<float>()});
      } else if (item.is_object()) {
        params.sampling.logit_bias.push_back(
            {static_cast<llama_token>(opt<int32_t>(item, "token", 0)),
             opt<float>(item, "bias", 0.0f)});
      }
    }
  }
}

json tool_calls_json(const std::vector<common_chat_tool_call> &calls) {
  json result = json::array();
  for (const auto &tc : calls) {
    json item = {
        {"type", "function"},
        {"function", {{"name", tc.name}, {"arguments", tc.arguments}}},
    };
    if (!tc.id.empty()) {
      item["id"] = tc.id;
    }
    result.push_back(item);
  }
  return result;
}

json action_completion(const json &options) {
  require_context();

  std::vector<std::string> stop_words = string_array(options, "stop");
  std::vector<std::string> media_paths = string_array(options, "media_paths");

  int32_t chat_format = opt<int32_t>(options, "chat_format", 0);
  std::string generation_prompt = opt_string(options, "generation_prompt");
  std::string reasoning_format = opt_string(options, "reasoning_format", "none");
  std::string chat_parser = opt_string(options, "chat_parser");

  common_chat_params jinja_chat_params;
  bool has_jinja_chat_params = false;
  common_params params = g_ctx->params;
  apply_speculative_options_json(options, params);
  params.sampling.grammar = {};
  params.sampling.generation_prompt.clear();
  params.sampling.grammar_triggers.clear();
  params.sampling.preserved_tokens.clear();

  const std::string grammar = opt_string(options, "grammar");
  bool has_grammar = !grammar.empty();
  if (has_grammar) {
    params.sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, grammar};
  }

  const std::string json_schema = response_schema_string(options);
  apply_preserved_tokens(params, options);
  apply_grammar_trigger_options(params, options);

  if (options.contains("grammar_lazy")) {
    params.sampling.grammar_lazy = opt<bool>(options, "grammar_lazy", false);
  }

  if (options.contains("messages") && options.at("messages").is_array()) {
    const bool use_jinja = opt<bool>(options, "jinja", true);
    if (use_jinja) {
      const std::string tools_str =
          options.contains("tools") && options.at("tools").is_array()
              ? options.at("tools").dump()
              : "";
      jinja_chat_params =
          format_jinja_chat(options.at("messages"), options, json_schema,
                            reasoning_format);
      has_jinja_chat_params = true;
      params.prompt = jinja_chat_params.prompt;
      chat_format = static_cast<int32_t>(jinja_chat_params.format);
      generation_prompt = jinja_chat_params.generation_prompt;
      chat_parser = jinja_chat_params.parser;

      for (const auto &token : jinja_chat_params.preserved_tokens) {
        const auto ids = common_tokenize(g_ctx->ctx, token, false, true);
        if (ids.size() == 1) {
          params.sampling.preserved_tokens.insert(ids[0]);
        }
      }

      if (!has_grammar) {
        const auto grammar_type = !tools_str.empty()
                                      ? COMMON_GRAMMAR_TYPE_TOOL_CALLS
                                      : COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT;
        params.sampling.grammar = {grammar_type, jinja_chat_params.grammar};
        params.sampling.grammar_lazy = jinja_chat_params.grammar_lazy;
        for (const auto &trigger : jinja_chat_params.grammar_triggers) {
          append_chat_template_grammar_trigger(g_ctx->ctx, params.sampling,
                                               trigger);
        }
        has_grammar = true;
      }

      for (const auto &stop : jinja_chat_params.additional_stops) {
        stop_words.push_back(stop);
      }
    } else {
      params.prompt = g_ctx->getFormattedChat(options.at("messages").dump(),
                                              opt_string(options, "chat_template"));
    }
  } else {
    params.prompt = opt_string(options, "prompt");
  }

  if (params.prompt.empty()) {
    throw std::runtime_error("Prompt is required");
  }

  if (!has_grammar && !json_schema.empty()) {
    params.sampling.grammar = {
        COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT,
        json_schema_to_grammar(json::parse(json_schema))};
  }
  params.sampling.generation_prompt = generation_prompt;
  apply_reasoning_budget_json(options, g_ctx->ctx, params.sampling,
                              has_jinja_chat_params ? &jinja_chat_params
                                                    : nullptr);
  apply_sampling_options(params, options);

  auto completion = g_ctx->completion;
  completion->rewind();
  g_ctx->params.prompt = params.prompt;
  g_ctx->params.sampling = params.sampling;
  g_ctx->params.speculative = params.speculative;
  g_ctx->params.antiprompt = stop_words;
  g_ctx->params.n_predict = params.n_predict;
  g_ctx->params.n_ctx = params.n_ctx;
  g_ctx->params.n_batch = params.n_batch;
  g_ctx->params.ctx_shift = params.ctx_shift;
  completion->prefill_text =
      rnllama::utf8_sanitize(opt_string(options, "prefill_text"));

  if (!completion->initSampling()) {
    throw std::runtime_error("Failed to initialize sampling");
  }
  completion->loadPrompt(media_paths);
  if (completion->context_full) {
    return ok({{"chat_format", chat_format},
               {"tokens_evaluated", completion->num_prompt_tokens},
               {"tokens_predicted", 0},
               {"text", ""},
               {"context_full", true},
               {"truncated", completion->truncated},
               {"stopped_eos", false},
               {"stopped_words", false},
               {"stopped_limited", false},
               {"interrupted", false}});
  }

  completion->beginCompletion(
      chat_format, common_reasoning_format_from_name(reasoning_format),
      generation_prompt, chat_parser);

  json streamed_tokens = json::array();
  int token_count = 0;
  size_t sent_count = 0;
  const int max_tokens = params.n_predict < 0
                             ? std::numeric_limits<int>::max()
                             : params.n_predict;

  while (completion->has_next_token && token_count < max_tokens) {
    rnllama::completion_token_output token_output = completion->doCompletion();
    if (token_output.tok == -1) {
      break;
    }
    token_count++;
    if (completion->incomplete) {
      continue;
    }

    const std::string token_text = common_token_to_piece(g_ctx->ctx,
                                                         token_output.tok);
    size_t pos = std::min(sent_count, completion->generated_text.size());
    const std::string str_test = completion->generated_text.substr(pos);

    bool is_stop_full = false;
    size_t stop_pos = completion->findStoppingStrings(
        str_test, token_text.size(), rnllama::STOP_FULL);
    if (stop_pos != std::string::npos) {
      is_stop_full = true;
      completion->generated_text.erase(
          completion->generated_text.begin() + pos + stop_pos,
          completion->generated_text.end());
      pos = std::min(sent_count, completion->generated_text.size());
    } else {
      stop_pos = completion->findStoppingStrings(
          str_test, token_text.size(), rnllama::STOP_PARTIAL);
    }

    if (stop_pos == std::string::npos ||
        (!completion->has_next_token && !is_stop_full && stop_pos > 0)) {
      const std::string to_send = completion->generated_text.substr(pos);
      sent_count += to_send.size();
      streamed_tokens.push_back(to_send);
    }
  }

  if (token_count >= max_tokens) {
    completion->stopped_limit = true;
  }

  completion->endCompletion();

  const auto timings = llama_perf_context(g_ctx->ctx);
  const double predicted_n =
      static_cast<double>(completion->num_tokens_predicted);
  const double predicted_ms = completion->t_token_generation * 1e3;
  json result = ok({
      {"chat_format", chat_format},
      {"tokens_evaluated",
       completion->num_prompt_tokens + completion->num_tokens_predicted},
      {"tokens_predicted", completion->num_tokens_predicted},
      {"draft_tokens", completion->num_draft_tokens},
      {"draft_tokens_accepted", completion->num_draft_tokens_accepted},
      {"truncated", completion->truncated},
      {"context_full", completion->context_full},
      {"interrupted", false},
      {"text", completion->generated_text},
      {"stopped_eos", completion->stopped_eos},
      {"stopped_words", completion->stopped_word},
      {"stopping_word", completion->stopping_word},
      {"stopped_limited", completion->stopped_limit},
      {"tokens", streamed_tokens},
      {"timings",
       {{"prompt_n", timings.n_p_eval},
        {"prompt_ms", timings.t_p_eval_ms},
        {"prompt_per_token_ms",
         timings.n_p_eval > 0 ? timings.t_p_eval_ms / timings.n_p_eval : 0.0},
        {"prompt_per_second",
         timings.t_p_eval_ms > 0
             ? 1e3 / timings.t_p_eval_ms * timings.n_p_eval
             : 0.0},
        {"predicted_n", predicted_n},
        {"predicted_ms", predicted_ms},
        {"predicted_per_token_ms",
         predicted_n > 0 ? predicted_ms / predicted_n : 0.0},
        {"predicted_per_second",
         predicted_ms > 0 ? 1e3 / predicted_ms * predicted_n : 0.0}}},
  });

  try {
    const auto final_output = completion->parseChatOutput(false);
    if (!final_output.reasoning_content.empty()) {
      result["reasoning_content"] = final_output.reasoning_content;
    }
    if (!final_output.content.empty()) {
      result["content"] = final_output.content;
    }
    const auto calls = tool_calls_json(final_output.tool_calls);
    if (!calls.empty()) {
      result["tool_calls"] = calls;
    }
  } catch (...) {
  }

  if (params.sampling.n_probs > 0 &&
      !completion->generated_token_probs.empty()) {
    result["completion_probabilities"] =
        token_probs_json(g_ctx->ctx, completion->generated_token_probs);
  }

  return result;
}

json action_tokenize(const json &payload) {
  require_context();
  const auto result =
      g_ctx->tokenize(opt_string(payload, "text"), string_array(payload, "media_paths"));
  json tokens = json::array();
  for (const auto token : result.tokens) {
    tokens.push_back(static_cast<int32_t>(token));
  }
  return ok({{"tokens", tokens},
             {"has_media", result.has_media},
             {"bitmap_hashes", result.bitmap_hashes},
             {"chunk_pos", result.chunk_pos},
             {"chunk_pos_media", result.chunk_pos_media}});
}

json action_detokenize(const json &payload) {
  require_context();
  std::vector<int32_t> tokens;
  if (payload.contains("tokens") && payload.at("tokens").is_array()) {
    for (const auto &token : payload.at("tokens")) {
      tokens.push_back(token.get<int32_t>());
    }
  }
  return ok({{"text", rnllama::tokens_to_str(g_ctx->ctx, tokens.begin(),
                                             tokens.end())}});
}

json action_save_session(const json &payload) {
  require_context();
  const std::string path = opt_string(payload, "path", "/session.bin");
  if (g_ctx->completion && !g_ctx->completion->embd.empty()) {
    auto &tokens = g_ctx->completion->embd;
    if (!llama_state_save_file(g_ctx->ctx, path.c_str(), tokens.data(),
                               tokens.size())) {
      throw std::runtime_error("Failed to save session");
    }
  } else if (!llama_state_save_file(g_ctx->ctx, path.c_str(), nullptr, 0)) {
    throw std::runtime_error("Failed to save session");
  }
  return ok();
}

json action_load_session(const json &payload) {
  require_context();
  const std::string path = opt_string(payload, "path", "/session.bin");
  std::vector<llama_token> tokens(g_ctx->n_ctx);
  size_t count = 0;
  if (!llama_state_load_file(g_ctx->ctx, path.c_str(), tokens.data(),
                             tokens.size(), &count)) {
    throw std::runtime_error("Failed to load session");
  }
  tokens.resize(count);
  g_ctx->completion->embd = std::move(tokens);
  g_ctx->completion->n_past = static_cast<llama_pos>(count);
  return ok({{"tokens_loaded", count}});
}

json action_formatted_chat(const json &payload) {
  require_context();
  const json messages = payload.contains("messages") ? payload.at("messages")
                                                     : json::array();
  json params = payload.contains("params") ? payload.at("params")
                                           : json::object();
  if (!params.contains("chat_template")) {
    const std::string tmpl = opt_string(payload, "template");
    if (!tmpl.empty()) {
      params["chat_template"] = tmpl;
    }
  }
  if (opt<bool>(params, "jinja", true)) {
    const auto chat_params =
        format_jinja_chat(messages, params, response_schema_string(params),
                          opt_string(params, "reasoning_format", "none"));
    json triggers = json::array();
    for (const auto &trigger : chat_params.grammar_triggers) {
      triggers.push_back({{"type", trigger.type},
                          {"value", trigger.value},
                          {"token", trigger.token}});
    }
    return ok({{"type", "jinja"},
               {"prompt", chat_params.prompt},
               {"chat_format", static_cast<int32_t>(chat_params.format)},
               {"grammar", chat_params.grammar},
               {"grammar_lazy", chat_params.grammar_lazy},
               {"grammar_triggers", triggers},
               {"generation_prompt", chat_params.generation_prompt},
               {"thinking_forced_open", false},
               {"thinking_start_tag", chat_params.thinking_start_tag},
               {"thinking_end_tag", chat_params.thinking_end_tag},
               {"preserved_tokens", chat_params.preserved_tokens},
               {"additional_stops", chat_params.additional_stops},
               {"chat_parser", chat_params.parser}});
  }
  return ok({{"type", "llama-chat"},
             {"prompt", g_ctx->getFormattedChat(messages.dump(),
                                                opt_string(payload, "template"))}});
}

json action_embedding(const json &payload) {
  require_context();
  common_params params = g_ctx->params;
  params.prompt = opt_string(payload, "text");
  params.embd_normalize = opt<int32_t>(payload, "embd_normalize",
                                       params.embd_normalize);
  g_ctx->params.prompt = params.prompt;
  const auto values = g_ctx->completion->embedding(params);
  return ok({{"embedding", values}});
}

json action_rerank(const json &payload) {
  require_context();
  std::vector<std::string> documents = string_array(payload, "documents");
  const auto scores =
      g_ctx->completion->rerank(opt_string(payload, "query"), documents);
  json results = json::array();
  for (size_t i = 0; i < scores.size(); i++) {
    results.push_back({{"index", static_cast<int32_t>(i)}, {"score", scores[i]}});
  }
  return ok({{"results", results}});
}

std::vector<common_adapter_lora_info> lora_adapters_from_json(
    const json &payload) {
  std::vector<common_adapter_lora_info> lora;
  if (!payload.contains("adapters") || !payload.at("adapters").is_array()) {
    throw std::runtime_error("adapters must be an array");
  }

  for (const auto &item : payload.at("adapters")) {
    if (!item.is_object()) {
      throw std::runtime_error("adapters must contain objects");
    }

    common_adapter_lora_info adapter;
    adapter.path = opt_string(item, "path");
    if (adapter.path.empty()) {
      throw std::runtime_error("LoRA adapter path is required");
    }
    adapter.scale = opt<float>(item, "scaled", opt<float>(item, "scale", 1.0f));
    lora.push_back(std::move(adapter));
  }

  return lora;
}

json loaded_lora_adapters_json() {
  require_context();

  json adapters = json::array();
  for (const auto &adapter : g_ctx->getLoadedLoraAdapters()) {
    adapters.push_back({{"path", adapter.path}, {"scaled", adapter.scale}});
  }
  return adapters;
}

json action_apply_lora_adapters(const json &payload) {
  require_context();
  g_ctx->applyLoraAdapters(lora_adapters_from_json(payload));
  return ok({{"lora_adapters", loaded_lora_adapters_json()}});
}

json action_remove_lora_adapters() {
  require_context();
  g_ctx->removeLoraAdapters();
  return ok({{"lora_adapters", loaded_lora_adapters_json()}});
}

json bench_result_json(const std::string &raw) {
  const auto parsed = json::parse(raw);
  if (!parsed.is_object() || parsed.empty()) {
    throw std::runtime_error("Benchmark failed");
  }

  return ok({{"nKvMax", parsed.at("n_kv_max")},
             {"nBatch", parsed.at("n_batch")},
             {"nUBatch", parsed.at("n_ubatch")},
             {"flashAttn", parsed.at("flash_attn")},
             {"isPpShared", parsed.at("is_pp_shared").get<int>() != 0},
             {"nGpuLayers", parsed.at("n_gpu_layers")},
             {"nThreads", parsed.at("n_threads")},
             {"nThreadsBatch", parsed.at("n_threads_batch")},
             {"pp", parsed.at("pp")},
             {"tg", parsed.at("tg")},
             {"pl", parsed.at("pl")},
             {"nKv", parsed.at("n_kv")},
             {"tPp", parsed.at("t_pp")},
             {"speedPp", parsed.at("speed_pp")},
             {"tTg", parsed.at("t_tg")},
             {"speedTg", parsed.at("speed_tg")},
             {"t", parsed.at("t")},
             {"speed", parsed.at("speed")}});
}

json action_bench(const json &payload) {
  require_context();
  if (!g_ctx->completion) {
    throw std::runtime_error("Completion context not initialized");
  }

  const int pp = opt<int32_t>(payload, "pp", 0);
  const int tg = opt<int32_t>(payload, "tg", 0);
  const int pl = opt<int32_t>(payload, "pl", 0);
  const int nr = opt<int32_t>(payload, "nr", 0);
  g_ctx->clearCache(false);
  return bench_result_json(g_ctx->completion->bench(pp, tg, pl, nr));
}

json multimodal_support_json() {
  require_context();
  if (!g_ctx->isMultimodalEnabled()) {
    return {{"vision", false}, {"audio", false}};
  }
  return {{"vision", g_ctx->isMultimodalSupportVision()},
          {"audio", g_ctx->isMultimodalSupportAudio()}};
}

json action_init_multimodal(const json &payload) {
  require_context();
  const std::string path = opt_string(payload, "path");
  if (path.empty()) {
    throw std::runtime_error("mmproj path is required");
  }

  g_ctx->params.ctx_shift = false;
  const bool ok_init =
      g_ctx->initMultimodal(path, opt<bool>(payload, "use_gpu", false),
                            opt<int32_t>(payload, "image_min_tokens", -1),
                            opt<int32_t>(payload, "image_max_tokens", -1));
  if (!ok_init) {
    throw std::runtime_error("Failed to initialize multimodal context");
  }
  return ok({{"support", multimodal_support_json()}});
}

} // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
const char *llama_node_wasm_start() {
  try {
    if (!g_started) {
      backend_init(false);
      g_started = true;
    }
    return set_result(ok());
  } catch (const std::exception &e) {
    return set_result(error_json(e.what()));
  }
}

EMSCRIPTEN_KEEPALIVE
const char *llama_node_wasm_action(const char *action_c,
                                   const char *payload_c) {
  try {
    const std::string action = action_c == nullptr ? "" : action_c;
    const json payload = payload_c == nullptr || payload_c[0] == '\0'
                             ? json::object()
                             : json::parse(payload_c);

    if (action == "load") {
      return set_result(action_load(payload));
    }
    if (action == "model_info") {
      return set_result(ok({{"modelInfo", model_info_json()}}));
    }
    if (action == "completion") {
      return set_result(action_completion(payload));
    }
    if (action == "tokenize") {
      return set_result(action_tokenize(payload));
    }
    if (action == "detokenize") {
      return set_result(action_detokenize(payload));
    }
    if (action == "save_session") {
      return set_result(action_save_session(payload));
    }
    if (action == "load_session") {
      return set_result(action_load_session(payload));
    }
    if (action == "formatted_chat") {
      return set_result(action_formatted_chat(payload));
    }
    if (action == "embedding") {
      return set_result(action_embedding(payload));
    }
    if (action == "rerank") {
      return set_result(action_rerank(payload));
    }
    if (action == "apply_lora_adapters") {
      return set_result(action_apply_lora_adapters(payload));
    }
    if (action == "remove_lora_adapters") {
      return set_result(action_remove_lora_adapters());
    }
    if (action == "get_loaded_lora_adapters") {
      require_context();
      return set_result(ok({{"lora_adapters", loaded_lora_adapters_json()}}));
    }
    if (action == "bench") {
      return set_result(action_bench(payload));
    }
    if (action == "init_multimodal") {
      return set_result(action_init_multimodal(payload));
    }
    if (action == "is_multimodal_enabled") {
      require_context();
      return set_result(ok({{"enabled", g_ctx->isMultimodalEnabled()}}));
    }
    if (action == "get_multimodal_support") {
      return set_result(ok({{"support", multimodal_support_json()}}));
    }
    if (action == "release_multimodal") {
      require_context();
      g_ctx->releaseMultimodal();
      return set_result(ok());
    }
    if (action == "clear_cache") {
      require_context();
      g_ctx->clearCache(opt<bool>(payload, "clear_data", false));
      return set_result(ok());
    }
    if (action == "release") {
      g_ctx.reset();
      return set_result(ok());
    }
    return set_result(error_json("Unknown action: " + action));
  } catch (const std::exception &e) {
    return set_result(error_json(e.what()));
  }
}

EMSCRIPTEN_KEEPALIVE
const char *llama_node_wasm_action_sync(const char *action_c,
                                        const char *payload_c) {
  return llama_node_wasm_action(action_c, payload_c);
}

EMSCRIPTEN_KEEPALIVE
const char *llama_node_wasm_release_all() {
  try {
    g_ctx.reset();
    if (g_started) {
      llama_backend_free();
      g_started = false;
    }
    return set_result(ok());
  } catch (const std::exception &e) {
    return set_result(error_json(e.what()));
  }
}

}
