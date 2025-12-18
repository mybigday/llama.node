// Parallel decoding methods implementation for LlamaContext

#include "LlamaContext.h"
#include "common.hpp"
#include "rn-llama/rn-llama.h"
#include "rn-llama/rn-completion.h"
#include "rn-llama/rn-slot.h"
#include "rn-llama/rn-slot-manager.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include <nlohmann/json.hpp>
#include <napi.h>

using json = nlohmann::ordered_json;

// EnableParallelMode(params: { n_parallel: number, n_batch?: number }): boolean
Napi::Value LlamaContext::EnableParallelMode(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (!_rn_ctx) {
    Napi::TypeError::New(env, "Context is disposed").ThrowAsJavaScriptException();
    return Napi::Boolean::New(env, false);
  }

  // Get parameters
  auto params = info[0].As<Napi::Object>();
  int32_t n_parallel = get_option<int32_t>(params, "n_parallel", 2);
  int32_t n_batch = get_option<int32_t>(params, "n_batch", 512);

  try {
    _rn_ctx->enableParallelMode(n_parallel, n_batch);

    // Start the processing loop after enabling parallel mode
    if (_rn_ctx->parallel_mode_enabled && _rn_ctx->slot_manager != nullptr) {
      _rn_ctx->slot_manager->start_processing_loop();
    }

    return Napi::Boolean::New(env, true);
  } catch (const std::exception& e) {
    Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    return Napi::Boolean::New(env, false);
  }
}

// DisableParallelMode(): void
void LlamaContext::DisableParallelMode(const Napi::CallbackInfo &info) {
  if (_rn_ctx) {
    _rn_ctx->disableParallelMode();
  }
}

// QueueCompletion(params: object): { requestId: number }
Napi::Value LlamaContext::QueueCompletion(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (!_rn_ctx) {
    Napi::TypeError::New(env, "Context is disposed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  if (!_rn_ctx->parallel_mode_enabled) {
    Napi::TypeError::New(env, "Parallel mode is not enabled. Call enableParallelMode() first.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  auto options = info[0].As<Napi::Object>();

  // Parse all the completion parameters similar to Completion()
  std::vector<std::string> stop_words;
  if (options.Has("stop") && options.Get("stop").IsArray()) {
    auto stop_words_array = options.Get("stop").As<Napi::Array>();
    for (size_t i = 0; i < stop_words_array.Length(); i++) {
      stop_words.push_back(stop_words_array.Get(i).ToString().Utf8Value());
    }
  }

  std::vector<std::string> media_paths;
  if (options.Has("media_paths")) {
    if (options.Get("media_paths").IsArray()) {
      auto media_paths_array = options.Get("media_paths").As<Napi::Array>();
      for (size_t i = 0; i < media_paths_array.Length(); i++) {
        media_paths.push_back(media_paths_array.Get(i).ToString().Utf8Value());
      }
    } else if (options.Get("media_paths").IsString()) {
      media_paths.push_back(options.Get("media_paths").ToString().Utf8Value());
    }
  }

  // Check if multimodal is enabled when media_paths are provided
  if (!media_paths.empty() && !(_rn_ctx->has_multimodal && _rn_ctx->mtmd_wrapper != nullptr)) {
    Napi::Error::New(env, "Multimodal support must be enabled via "
                          "initMultimodal to use media_paths")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  int32_t chat_format = get_option<int32_t>(options, "chat_format", 0);
  bool thinking_forced_open = get_option<bool>(options, "thinking_forced_open", false);
  std::string reasoning_format = get_option<std::string>(options, "reasoning_format", "none");

  // Parse parameters
  common_params params = _rn_ctx->params;
  auto grammar_from_params = get_option<std::string>(options, "grammar", "");
  auto has_grammar_set = !grammar_from_params.empty();
  if (has_grammar_set) {
    params.sampling.grammar = grammar_from_params;
  }

  std::string json_schema_str = "";
  if (options.Has("response_format")) {
    auto response_format = options.Get("response_format").As<Napi::Object>();
    auto response_format_type =
        get_option<std::string>(response_format, "type", "text");
    if (response_format_type == "json_schema" &&
        response_format.Has("json_schema")) {
      auto json_schema = response_format.Get("json_schema").As<Napi::Object>();
      json_schema_str =
          json_schema.Has("schema")
              ? json_stringify(json_schema.Get("schema").As<Napi::Object>())
              : "{}";
    } else if (response_format_type == "json_object") {
      json_schema_str =
          response_format.Has("schema")
              ? json_stringify(response_format.Get("schema").As<Napi::Object>())
              : "{}";
    }
  }

  // Handle preserved_tokens from options
  if (options.Has("preserved_tokens")) {
    auto preserved_tokens = options.Get("preserved_tokens").As<Napi::Array>();
    for (size_t i = 0; i < preserved_tokens.Length(); i++) {
      auto token = preserved_tokens.Get(i).ToString().Utf8Value();
      auto ids =
          common_tokenize(_rn_ctx->ctx, token, /* add_special= */ false,
                          /* parse_special= */ true);
      if (ids.size() == 1) {
        params.sampling.preserved_tokens.insert(ids[0]);
      }
    }
  }

  // Handle grammar_triggers from options
  if (options.Has("grammar_triggers")) {
    auto grammar_triggers = options.Get("grammar_triggers").As<Napi::Array>();
    for (size_t i = 0; i < grammar_triggers.Length(); i++) {
      auto trigger_obj = grammar_triggers.Get(i).As<Napi::Object>();

      auto type = static_cast<common_grammar_trigger_type>(
          trigger_obj.Get("type").ToNumber().Int32Value());
      auto word = trigger_obj.Get("value").ToString().Utf8Value();

      if (type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
        auto ids =
            common_tokenize(_rn_ctx->ctx, word, /* add_special= */ false,
                            /* parse_special= */ true);
        if (ids.size() == 1) {
          auto token = ids[0];
          if (std::find(params.sampling.preserved_tokens.begin(),
                        params.sampling.preserved_tokens.end(),
                        (llama_token)token) ==
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
      } else {
        common_grammar_trigger trigger;
        trigger.type = type;
        trigger.value = word;
        if (type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
          auto token =
              (llama_token)trigger_obj.Get("token").ToNumber().Int32Value();
          trigger.token = token;
        }
        params.sampling.grammar_triggers.push_back(std::move(trigger));
      }
    }
  }

  // Handle grammar_lazy from options
  if (options.Has("grammar_lazy")) {
    params.sampling.grammar_lazy =
        options.Get("grammar_lazy").ToBoolean().Value();
  }

  std::string prompt = "";
  if (options.Has("messages") && options.Get("messages").IsArray()) {
    auto messages = options.Get("messages").As<Napi::Array>();
    auto chat_template = get_option<std::string>(options, "chat_template", "");
    auto jinja = get_option<bool>(options, "jinja", false);
    if (jinja) {
      auto tools_str =
          !is_nil(options.Get("tools"))
              ? json_stringify(options.Get("tools").As<Napi::Array>())
              : "";
      auto parallel_tool_calls =
          get_option<bool>(options, "parallel_tool_calls", false);
      auto tool_choice =
          get_option<std::string>(options, "tool_choice", "none");
      auto enable_thinking = get_option<bool>(options, "enable_thinking", true);
      auto add_generation_prompt = get_option<bool>(options, "add_generation_prompt", true);
      auto now_str = get_option<std::string>(options, "now", "");

      std::map<std::string, std::string> chat_template_kwargs;
      if (options.Has("chat_template_kwargs") && options.Get("chat_template_kwargs").IsObject()) {
        auto kwargs_obj = options.Get("chat_template_kwargs").As<Napi::Object>();
        auto props = kwargs_obj.GetPropertyNames();
        for (uint32_t i = 0; i < props.Length(); i++) {
          auto key = props.Get(i).ToString().Utf8Value();
          auto val = kwargs_obj.Get(key).ToString().Utf8Value();
          chat_template_kwargs[key] = val;
        }
      }

      common_chat_params chatParams;

      try {
        chatParams = _rn_ctx->getFormattedChatWithJinja(
            json_stringify(messages), chat_template,
            json_schema_str, tools_str, parallel_tool_calls, tool_choice, enable_thinking,
            add_generation_prompt, now_str, chat_template_kwargs);
      } catch (const std::exception &e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Undefined();
      }

      prompt = chatParams.prompt;

      chat_format = chatParams.format;
      thinking_forced_open = chatParams.thinking_forced_open;

      for (const auto &token : chatParams.preserved_tokens) {
        auto ids =
            common_tokenize(_rn_ctx->ctx, token, /* add_special= */ false,
                            /* parse_special= */ true);
        if (ids.size() == 1) {
          params.sampling.preserved_tokens.insert(ids[0]);
        }
      }

      if (!has_grammar_set) {
        // grammar param always wins jinja template & json_schema
        params.sampling.grammar = chatParams.grammar;
        params.sampling.grammar_lazy = chatParams.grammar_lazy;
        for (const auto &trigger : chatParams.grammar_triggers) {
          params.sampling.grammar_triggers.push_back(trigger);
        }
        has_grammar_set = true;
      }

      for (const auto &stop : chatParams.additional_stops) {
        stop_words.push_back(stop);
      }
    } else {
      auto formatted = _rn_ctx->getFormattedChat(
          json_stringify(messages), chat_template);
      prompt = formatted;
    }
  } else {
    prompt = get_option<std::string>(options, "prompt", "");
  }
  if (prompt.empty()) {
    Napi::TypeError::New(env, "Prompt is required")
        .ThrowAsJavaScriptException();
  }

  if (!has_grammar_set && !json_schema_str.empty()) {
    params.sampling.grammar =
        json_schema_to_grammar(json::parse(json_schema_str));
  }

  std::string prefill_text = get_option<std::string>(options, "prefill_text", "");

  // Handle state management parameters
  std::string load_state_path = get_option<std::string>(options, "load_state_path", "");
  std::string save_state_path = get_option<std::string>(options, "save_state_path", "");
  int32_t load_state_size = get_option<int32_t>(options, "load_state_size", -1);
  int32_t save_state_size = get_option<int32_t>(options, "save_state_size", -1);

  // ALL Sampling parameters
  params.n_predict = get_option<int32_t>(options, "n_predict", -1);
  params.sampling.temp = get_option<float>(options, "temperature", 0.80f);
  params.sampling.top_k = get_option<int32_t>(options, "top_k", 40);
  params.sampling.top_p = get_option<float>(options, "top_p", 0.95f);
  params.sampling.min_p = get_option<float>(options, "min_p", 0.05f);
  params.sampling.mirostat = get_option<int32_t>(options, "mirostat", 0.00f);
  params.sampling.mirostat_tau =
      get_option<float>(options, "mirostat_tau", 5.00f);
  params.sampling.mirostat_eta =
      get_option<float>(options, "mirostat_eta", 0.10f);
  params.sampling.penalty_last_n =
      get_option<int32_t>(options, "penalty_last_n", 64);
  params.sampling.penalty_repeat =
      get_option<float>(options, "penalty_repeat", 1.00f);
  params.sampling.penalty_freq =
      get_option<float>(options, "penalty_freq", 0.00f);
  params.sampling.penalty_present =
      get_option<float>(options, "penalty_present", 0.00f);
  params.sampling.typ_p = get_option<float>(options, "typical_p", 1.00f);
  params.sampling.xtc_threshold =
      get_option<float>(options, "xtc_threshold", 0.00f);
  params.sampling.xtc_probability =
      get_option<float>(options, "xtc_probability", 0.10f);
  params.sampling.dry_multiplier =
      get_option<float>(options, "dry_multiplier", 1.75f);
  params.sampling.dry_base = get_option<float>(options, "dry_base", 2);
  params.sampling.dry_allowed_length =
      get_option<float>(options, "dry_allowed_length", -1);
  params.sampling.dry_penalty_last_n =
      get_option<float>(options, "dry_penalty_last_n", 0);
  params.sampling.top_n_sigma =
      get_option<float>(options, "top_n_sigma", -1.0f);
  params.sampling.ignore_eos = get_option<bool>(options, "ignore_eos", false);
  params.n_keep = get_option<int32_t>(options, "n_keep", 0);
  params.sampling.seed =
      get_option<int32_t>(options, "seed", LLAMA_DEFAULT_SEED);
  params.sampling.n_probs = get_option<int32_t>(options, "n_probs", 0);

  // DRY sequence breakers
  if (options.Has("dry_sequence_breakers") && options.Get("dry_sequence_breakers").IsArray()) {
    auto dry_array = options.Get("dry_sequence_breakers").As<Napi::Array>();
    params.sampling.dry_sequence_breakers.clear();
    for (size_t i = 0; i < dry_array.Length(); i++) {
      params.sampling.dry_sequence_breakers.push_back(dry_array.Get(i).ToString().Utf8Value());
    }
  }

  // Logit bias
  if (options.Has("logit_bias") && options.Get("logit_bias").IsArray()) {
    auto logit_bias_array = options.Get("logit_bias").As<Napi::Array>();
    params.sampling.logit_bias.clear();
    for (size_t i = 0; i < logit_bias_array.Length(); i++) {
      auto bias_pair = logit_bias_array.Get(i).As<Napi::Array>();
      if (bias_pair.Length() == 2) {
        llama_token token = bias_pair.Get(static_cast<uint32_t>(0)).ToNumber().Int32Value();
        float bias = bias_pair.Get(static_cast<uint32_t>(1)).ToNumber().FloatValue();
        params.sampling.logit_bias[token].bias = bias;
      }
    }
  }

  // Stop words assignment to params
  params.antiprompt = stop_words;

  // Handle reasoning format
  common_reasoning_format reasoning_format_enum = common_reasoning_format_from_name(reasoning_format);

  // Tokenize prompt
  auto tokenize_result = _rn_ctx->tokenize(prompt, media_paths);

  // Create callback wrapper
  Napi::ThreadSafeFunction tsfn;
  bool hasCallback = info.Length() > 1 && info[1].IsFunction();

  if (hasCallback) {
    tsfn = Napi::ThreadSafeFunction::New(
      env,
      info[1].As<Napi::Function>(),
      "QueueCompletionCallback",
      0,
      1
    );
  }

  // Capture validity flag and slot_manager to prevent use-after-free
  auto context_valid = _context_valid;
  auto slot_manager = _rn_ctx->slot_manager;

  // Queue the request
  int32_t requestId = _rn_ctx->slot_manager->queue_request(
    params,
    tokenize_result.tokens,
    media_paths,
    prompt,
    chat_format,
    reasoning_format_enum,
    thinking_forced_open,
    prefill_text,
    load_state_path,
    save_state_path,
    load_state_size,
    save_state_size,
    [tsfn, hasCallback, chat_format, thinking_forced_open, context_valid, slot_manager](const completion_token_output& token) {
      if (!hasCallback) return;

      struct TokenData {
        completion_token_output token;
        int32_t chat_format;
        bool thinking_forced_open;
        std::string accumulated_text;
        std::string content;
        std::string reasoning_content;
        std::vector<common_chat_tool_call> tool_calls;
      };

      auto callback = [](Napi::Env env, Napi::Function jsCallback, TokenData* data) {
        Napi::Object result = Napi::Object::New(env);
        result.Set("requestId", Napi::Number::New(env, data->token.request_id));
        result.Set("token", Napi::String::New(env, data->token.text));

        if (!data->token.probs.empty()) {
          Napi::Array probs = Napi::Array::New(env);
          for (size_t i = 0; i < data->token.probs.size(); i++) {
            Napi::Object prob = Napi::Object::New(env);
            prob.Set("tok", Napi::Number::New(env, data->token.probs[i].tok));
            prob.Set("prob", Napi::Number::New(env, data->token.probs[i].prob));
            probs.Set(i, prob);
          }
          result.Set("probs", probs);
        }

        // Add chat format metadata
        if (data->chat_format > 0) {
          result.Set("chat_format", Napi::Number::New(env, data->chat_format));

          // Add parsed content if available
          if (!data->content.empty()) {
            result.Set("content", Napi::String::New(env, data->content));
          }
          if (!data->reasoning_content.empty()) {
            result.Set("reasoning_content", Napi::String::New(env, data->reasoning_content));
          }
          if (!data->tool_calls.empty()) {
            Napi::Array tool_calls = Napi::Array::New(env);
            for (size_t i = 0; i < data->tool_calls.size(); i++) {
              const auto &tc = data->tool_calls[i];
              Napi::Object tool_call = Napi::Object::New(env);
              tool_call.Set("type", "function");
              Napi::Object function = Napi::Object::New(env);
              function.Set("name", tc.name);
              function.Set("arguments", tc.arguments);
              tool_call.Set("function", function);
              if (!tc.id.empty()) {
                tool_call.Set("id", tc.id);
              }
              tool_calls.Set(i, tool_call);
            }
            result.Set("tool_calls", tool_calls);
          }
        }

        // Always use consistent callback format with error as first parameter
        jsCallback.Call({env.Null(), result});
        delete data;
      };

      auto* data = new TokenData;
      data->token = token;
      data->chat_format = chat_format;
      data->thinking_forced_open = thinking_forced_open;

      // For chat format, try to parse partial output
      // Check context validity to prevent use-after-free
      if (chat_format > 0 && context_valid && context_valid->load() && slot_manager != nullptr) {
        // Get the slot for this request to access accumulated text
        auto slot = slot_manager->get_slot_by_request_id(token.request_id);
        if (slot != nullptr) {
          try {
            // Use slot's own parseChatOutput method
            auto partial_output = slot->parseChatOutput(true);

            data->accumulated_text = partial_output.accumulated_text;
            data->content = partial_output.content;
            data->reasoning_content = partial_output.reasoning_content;
            data->tool_calls = partial_output.tool_calls;
          } catch (const std::exception &e) {
            // Silently ignore parse errors for partial output
          }
        }
      }

      auto status = tsfn.BlockingCall(data, callback);
      if (status != napi_ok) {
        delete data;
      }
    },
    [tsfn, hasCallback](llama_rn_slot* slot) {
      if (!hasCallback) return;

      struct CompletionResult {
        int32_t request_id;
        std::string text;
        std::string content;
        std::string reasoning_content;
        std::vector<common_chat_tool_call> tool_calls;
        bool stopped_eos;
        bool stopped_limit;
        bool stopped_word;
        bool context_full;
        int32_t chat_format;
        bool thinking_forced_open;
        size_t tokens_evaluated;
        size_t tokens_predicted;
        rnllama::slot_timings timings;
      };

      // Parse chat output if chat format is enabled
      std::string content;
      std::string reasoning_content;
      std::vector<common_chat_tool_call> tool_calls;

      if (slot->current_chat_format > 0) {
        try {
          // Use slot's own parseChatOutput method
          auto final_output = slot->parseChatOutput(false);

          content = final_output.content;
          reasoning_content = final_output.reasoning_content;
          tool_calls = final_output.tool_calls;
        } catch (const std::exception &e) {
          // Silently ignore parse errors for now - we still have the raw text
        }
      }

      // Get timings from slot
      rnllama::slot_timings slot_timings = slot->get_timings();

      auto* result_data = new CompletionResult{
        slot->request_id,
        slot->generated_text,
        content,
        reasoning_content,
        tool_calls,
        slot->stopped_eos,
        slot->stopped_limit,
        slot->stopped_word,
        slot->context_full,
        slot->current_chat_format,
        slot->current_thinking_forced_open,
        static_cast<size_t>(slot->n_decoded),
        slot->num_tokens_predicted,
        slot_timings
      };

      auto callback = [](Napi::Env env, Napi::Function jsCallback, CompletionResult* data) {
        Napi::Object result = Napi::Object::New(env);
        result.Set("requestId", Napi::Number::New(env, data->request_id));
        result.Set("text", Napi::String::New(env, data->text));
        result.Set("stopped_eos", Napi::Boolean::New(env, data->stopped_eos));
        result.Set("stopped_limit", Napi::Boolean::New(env, data->stopped_limit));
        result.Set("stopped_word", Napi::Boolean::New(env, data->stopped_word));
        result.Set("context_full", Napi::Boolean::New(env, data->context_full));
        result.Set("tokens_evaluated", Napi::Number::New(env, data->tokens_evaluated));
        result.Set("tokens_predicted", Napi::Number::New(env, data->tokens_predicted));
        result.Set("chat_format", Napi::Number::New(env, data->chat_format));

        // Add parsed content if available
        if (!data->content.empty()) {
          result.Set("content", Napi::String::New(env, data->content));
        }

        if (!data->reasoning_content.empty()) {
          result.Set("reasoning_content", Napi::String::New(env, data->reasoning_content));
        }

        // Convert tool calls to JavaScript format
        if (!data->tool_calls.empty()) {
          Napi::Array tool_calls = Napi::Array::New(env);
          for (size_t i = 0; i < data->tool_calls.size(); i++) {
            const auto &tc = data->tool_calls[i];
            Napi::Object tool_call = Napi::Object::New(env);
            tool_call.Set("type", "function");
            Napi::Object function = Napi::Object::New(env);
            function.Set("name", tc.name);
            function.Set("arguments", tc.arguments);
            tool_call.Set("function", function);
            if (!tc.id.empty()) {
              tool_call.Set("id", tc.id);
            }
            tool_calls.Set(i, tool_call);
          }
          result.Set("tool_calls", tool_calls);
        }

        // Add timings
        Napi::Object timingsObj = Napi::Object::New(env);
        timingsObj.Set("cache_n", Napi::Number::New(env, data->timings.cache_n));
        timingsObj.Set("prompt_n", Napi::Number::New(env, data->timings.prompt_n));
        timingsObj.Set("prompt_ms", Napi::Number::New(env, data->timings.prompt_ms));
        timingsObj.Set("prompt_per_token_ms", Napi::Number::New(env, data->timings.prompt_per_token_ms));
        timingsObj.Set("prompt_per_second", Napi::Number::New(env, data->timings.prompt_per_second));
        timingsObj.Set("predicted_n", Napi::Number::New(env, data->timings.predicted_n));
        timingsObj.Set("predicted_ms", Napi::Number::New(env, data->timings.predicted_ms));
        timingsObj.Set("predicted_per_token_ms", Napi::Number::New(env, data->timings.predicted_per_token_ms));
        timingsObj.Set("predicted_per_second", Napi::Number::New(env, data->timings.predicted_per_second));
        result.Set("timings", timingsObj);

        jsCallback.Call({env.Null(), result});
        delete data;
      };

      auto status = tsfn.BlockingCall(result_data, callback);
      if (status != napi_ok) {
        delete result_data;
      }

      if (hasCallback) {
        tsfn.Release();
      }
    }
  );

  Napi::Object result = Napi::Object::New(env);
  result.Set("requestId", Napi::Number::New(env, requestId));
  return result;
}

// QueueEmbedding(text: string, params?: object): { requestId: number }
Napi::Value LlamaContext::QueueEmbedding(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (!_rn_ctx) {
    Napi::TypeError::New(env, "Context is disposed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  if (!_rn_ctx->parallel_mode_enabled) {
    Napi::TypeError::New(env, "Parallel mode is not enabled. Call enableParallelMode() first.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  std::string text = info[0].ToString().Utf8Value();

  auto params = Napi::Object::New(env);
  if (info.Length() >= 2 && info[1].IsObject()) {
    params = info[1].As<Napi::Object>();
  }

  int embd_normalize = get_option<int32_t>(params, "embd_normalize", 2);

  // Tokenize text
  const llama_vocab* vocab = llama_model_get_vocab(_rn_ctx->model);
  const bool add_bos = llama_vocab_get_add_bos(vocab);
  const bool is_enc_dec = llama_model_has_encoder(_rn_ctx->model);
  std::vector<llama_token> tokens = common_tokenize(
    _rn_ctx->ctx,
    text,
    add_bos || is_enc_dec,
    true
  );

  // Create callback wrapper
  Napi::ThreadSafeFunction tsfn;
  bool hasCallback = info.Length() > 2 && info[2].IsFunction();

  if (hasCallback) {
    tsfn = Napi::ThreadSafeFunction::New(
      env,
      info[2].As<Napi::Function>(),
      "QueueEmbeddingCallback",
      0,
      1
    );
  }

  // Queue embedding request
  int32_t requestId = _rn_ctx->slot_manager->queue_embedding_request(
    tokens,
    embd_normalize,
    [tsfn, hasCallback](int32_t requestId, const std::vector<float>& embedding) {
      if (!hasCallback) return;

      struct EmbeddingData {
        int32_t requestId;
        std::vector<float> embedding;
      };

      auto callback = [](Napi::Env env, Napi::Function jsCallback, EmbeddingData* data) {
        Napi::Object result = Napi::Object::New(env);
        result.Set("requestId", Napi::Number::New(env, data->requestId));

        Napi::Array embeddingArray = Napi::Array::New(env);
        for (size_t i = 0; i < data->embedding.size(); i++) {
          embeddingArray.Set(i, Napi::Number::New(env, data->embedding[i]));
        }
        result.Set("embedding", embeddingArray);

        jsCallback.Call({env.Null(), result});
        delete data;
      };

      auto* data = new EmbeddingData{requestId, embedding};
      auto status = tsfn.BlockingCall(data, callback);
      if (status != napi_ok) {
        delete data;
      }
      tsfn.Release();
    }
  );

  Napi::Object result = Napi::Object::New(env);
  result.Set("requestId", Napi::Number::New(env, requestId));
  return result;
}

// QueueRerank(query: string, documents: string[], params?: object): { requestId: number }
Napi::Value LlamaContext::QueueRerank(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (!_rn_ctx) {
    Napi::TypeError::New(env, "Context is disposed").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  if (!_rn_ctx->parallel_mode_enabled) {
    Napi::TypeError::New(env, "Parallel mode is not enabled. Call enableParallelMode() first.")
        .ThrowAsJavaScriptException();
    return env.Undefined();
  }

  std::string query = info[0].ToString().Utf8Value();
  auto documents_array = info[1].As<Napi::Array>();

  std::vector<std::string> documents;
  for (size_t i = 0; i < documents_array.Length(); i++) {
    documents.push_back(documents_array.Get(i).ToString().Utf8Value());
  }

  auto params = Napi::Object::New(env);
  if (info.Length() >= 3 && info[2].IsObject()) {
    params = info[2].As<Napi::Object>();
  }

  int normalize = get_option<int32_t>(params, "normalize", 0);

  // Create callback wrapper
  Napi::ThreadSafeFunction tsfn;
  bool hasCallback = info.Length() > 3 && info[3].IsFunction();

  if (hasCallback) {
    tsfn = Napi::ThreadSafeFunction::New(
      env,
      info[3].As<Napi::Function>(),
      "QueueRerankCallback",
      0,
      1
    );
  }

  // Queue rerank request
  int32_t requestId = _rn_ctx->slot_manager->queue_rerank_request(
    query,
    documents,
    normalize,
    [tsfn, hasCallback, documents](int32_t requestId, const std::vector<float>& scores) {
      if (!hasCallback) return;

      struct RerankData {
        int32_t requestId;
        std::vector<float> scores;
        std::vector<std::string> documents;
      };

      auto callback = [](Napi::Env env, Napi::Function jsCallback, RerankData* data) {
        Napi::Object result = Napi::Object::New(env);
        result.Set("requestId", Napi::Number::New(env, data->requestId));

        Napi::Array resultsArray = Napi::Array::New(env);
        for (size_t i = 0; i < data->scores.size(); i++) {
          Napi::Object item = Napi::Object::New(env);
          item.Set("score", Napi::Number::New(env, data->scores[i]));
          item.Set("index", Napi::Number::New(env, i));
          item.Set("document", Napi::String::New(env, data->documents[i]));
          resultsArray.Set(i, item);
        }
        result.Set("results", resultsArray);

        jsCallback.Call({env.Null(), result});
        delete data;
      };

      auto* data = new RerankData{requestId, scores, documents};
      auto status = tsfn.BlockingCall(data, callback);
      if (status != napi_ok) {
        delete data;
      }
      tsfn.Release();
    }
  );

  Napi::Object result = Napi::Object::New(env);
  result.Set("requestId", Napi::Number::New(env, requestId));
  return result;
}

// CancelRequest(requestId: number): void
void LlamaContext::CancelRequest(const Napi::CallbackInfo &info) {
  if (_rn_ctx && _rn_ctx->parallel_mode_enabled && _rn_ctx->slot_manager) {
    int32_t requestId = info[0].ToNumber().Int32Value();
    _rn_ctx->slot_manager->cancel_request(requestId);
  }
}