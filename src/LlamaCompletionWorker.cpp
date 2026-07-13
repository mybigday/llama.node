#include "LlamaCompletionWorker.h"
#include "LlamaContext.h"
#include "rn-llama/rn-completion.h"
#include <limits>

// Helper function to convert token probabilities to JavaScript format
Napi::Array TokenProbsToArray(Napi::Env env, llama_context* ctx, const std::vector<rnllama::completion_token_output>& probs) {
  Napi::Array result = Napi::Array::New(env);
  for (size_t i = 0; i < probs.size(); i++) {
    const auto &prob = probs[i];
    Napi::Object token_obj = Napi::Object::New(env);

    std::string token_str = rnllama::tokens_to_output_formatted_string(ctx, prob.tok);
    token_obj.Set("content", Napi::String::New(env, token_str));

    Napi::Array token_probs = Napi::Array::New(env);
    for (size_t j = 0; j < prob.probs.size(); j++) {
      const auto &p = prob.probs[j];
      Napi::Object prob_obj = Napi::Object::New(env);
      std::string tok_str = rnllama::tokens_to_output_formatted_string(ctx, p.tok);
      prob_obj.Set("tok_str", Napi::String::New(env, tok_str));
      prob_obj.Set("prob", Napi::Number::New(env, p.prob));
      token_probs.Set(j, prob_obj);
    }
    token_obj.Set("probs", token_probs);
    result.Set(i, token_obj);
  }
  return result;
}


LlamaCompletionWorker::LlamaCompletionWorker(
    const Napi::CallbackInfo &info, rnllama::llama_rn_context* rn_ctx,
    Napi::Function callback,
    common_params params,
    std::vector<std::string> stop_words,
    int32_t chat_format,
    const std::string &generation_prompt,
    std::string reasoning_format,
    const std::string &chat_parser,
    const std::vector<std::string> &media_paths,
    const std::vector<llama_token> &guide_tokens,
    bool has_vocoder,
    rnllama::tts_type tts_type_val,
    const std::string &prefill_text)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _rn_ctx(rn_ctx),
      _params(params), _stop_words(stop_words), _chat_format(chat_format),
      _generation_prompt(generation_prompt),
      _reasoning_format(reasoning_format),
      _chat_parser(chat_parser),
      _media_paths(media_paths), _guide_tokens(guide_tokens),
      _prefill_text(prefill_text),
      _has_vocoder(has_vocoder), _tts_type(tts_type_val) {
  if (!callback.IsEmpty()) {
    _tsfn = Napi::ThreadSafeFunction::New(info.Env(), callback,
                                          "LlamaCompletionCallback", 0, 1);
    _has_callback = true;
  }
}

LlamaCompletionWorker::~LlamaCompletionWorker() {
  if (_has_callback) {
    _tsfn.Release();
  }
}


void LlamaCompletionWorker::Execute() {
  try {
    // Check if vocab_only mode is enabled - if so, return empty result
    if (_params.vocab_only) {
      // Return empty completion result for vocab_only mode
      _result.tokens_evaluated = 0;
      _result.tokens_predicted = 0;
      _result.text = "";
      _result.stopped_limited = true;
      _result.truncated = false;
      _result.context_full = false;
      _result.stopped_eos = false;
      _result.stopped_words = false;
      if (_onComplete) {
        _onComplete();
      }
      return;
    }

    auto completion = _rn_ctx->completion;

    // Prepare completion context
    completion->rewind();

    // Set up parameters
    _rn_ctx->params.prompt = _params.prompt;
    _rn_ctx->params.sampling = _params.sampling;
    _rn_ctx->params.speculative = _params.speculative;
    _rn_ctx->params.antiprompt = _stop_words;
    _rn_ctx->params.n_predict = _params.n_predict;
    _rn_ctx->params.n_ctx = _params.n_ctx;
    _rn_ctx->params.n_batch = _params.n_batch;
    _rn_ctx->params.ctx_shift = _params.ctx_shift;

    // Set prefill text
    completion->prefill_text = rnllama::utf8_sanitize(_prefill_text);

    // Set up TTS guide tokens if enabled
    if (_has_vocoder && _rn_ctx->tts_wrapper != nullptr) {
      _rn_ctx->tts_wrapper->guide_tokens = _guide_tokens;
      _rn_ctx->tts_wrapper->next_token_uses_guide_token = true;
    }

    if (!_media_paths.empty() &&
        speculative_has_type(_params.speculative, COMMON_SPECULATIVE_TYPE_DRAFT_MTP)) {
      SetError("MTP speculative decoding currently supports text-only completion in llama.node");
      return;
    }

    // Initialize sampling
    if (!completion->initSampling()) {
      SetError("Failed to initialize sampling");
      return;
    }

    // Load prompt (handles both text-only and multimodal)
    completion->loadPrompt(_media_paths);

    // Check if context is full after loading prompt
    if (completion->context_full) {
      _result.context_full = true;
      return;
    }

    // Begin completion with chat format and reasoning settings
    completion->beginCompletion(
        _chat_format,
        common_reasoning_format_from_name(_reasoning_format),
        _generation_prompt,
        _chat_parser);

    // Main completion loop
    int token_count = 0;
    const int max_tokens = _params.n_predict < 0 ? std::numeric_limits<int>::max() : _params.n_predict;
    size_t sent_count = 0;
    while (completion->has_next_token && !_interrupted && token_count < max_tokens) {
      // Get next token using rn-llama completion
      rnllama::completion_token_output token_output = completion->doCompletion();

      if (token_output.tok == -1) {
        break;
      }

      token_count++;
      if (completion->incomplete) {
        continue;
      }

      std::string token_text = common_token_to_piece(_rn_ctx->ctx, token_output.tok);
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

        // Handle streaming callback
        if (!_has_callback) {
          continue;
        }

        struct TokenData {
          std::string token;
          std::string content;
          std::string reasoning_content;
          std::vector<common_chat_tool_call> tool_calls;
          std::string accumulated_text;
          std::vector<rnllama::completion_token_output> completion_probabilities;
          llama_context* ctx;
        };

        rnllama::completion_chat_output partial_output;
        try {
          partial_output = completion->parseChatOutput(true);
        } catch (const std::exception &) {
          partial_output.accumulated_text =
              completion->prefill_text + completion->generated_text;
        }

        // Extract completion probabilities if n_probs > 0, similar to iOS implementation
        std::vector<rnllama::completion_token_output> probs_output;
        if (_rn_ctx->params.sampling.n_probs > 0) {
          size_t probs_pos = std::min(_sent_token_probs_index, completion->generated_token_probs.size());
          size_t probs_stop_pos = completion->generated_token_probs.size();
          if (probs_pos < probs_stop_pos) {
            probs_output = std::vector<rnllama::completion_token_output>(
              completion->generated_token_probs.begin() + probs_pos,
              completion->generated_token_probs.begin() + probs_stop_pos
            );
          }
          _sent_token_probs_index = probs_stop_pos;
        }

        TokenData *token_data = new TokenData{
          to_send,
          partial_output.content,
          partial_output.reasoning_content,
          partial_output.tool_calls,
          partial_output.accumulated_text,
          probs_output,
          _rn_ctx->ctx
        };

        _tsfn.BlockingCall(token_data, [](Napi::Env env, Napi::Function jsCallback,
                                          TokenData *data) {
          auto obj = Napi::Object::New(env);
          obj.Set("token", Napi::String::New(env, data->token));
          if (!data->content.empty()) {
            obj.Set("content", Napi::String::New(env, data->content));
          }
          if (!data->reasoning_content.empty()) {
            obj.Set("reasoning_content", Napi::String::New(env, data->reasoning_content));
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
            obj.Set("tool_calls", tool_calls);
          }
          obj.Set("accumulated_text", Napi::String::New(env, data->accumulated_text));

          // Add completion_probabilities if available
          if (!data->completion_probabilities.empty()) {
            obj.Set("completion_probabilities", TokenProbsToArray(env, data->ctx, data->completion_probabilities));
          }

          delete data;
          jsCallback.Call({obj});
        });
      }
    }

    // Check stopping conditions
    if (token_count >= max_tokens) {
      _result.stopped_limited = true;
    } else if (!completion->has_next_token && completion->n_remain == 0) {
      _result.stopped_limited = true;
    }

    // Set completion results from rn-llama completion context
    // tokens_evaluated should include both prompt tokens and generated tokens that were processed
    _result.tokens_evaluated = completion->num_prompt_tokens + completion->num_tokens_predicted;
    _result.tokens_predicted = completion->num_tokens_predicted;
    _result.draft_tokens = completion->num_draft_tokens;
    _result.draft_tokens_accepted = completion->num_draft_tokens_accepted;
    _result.truncated = completion->truncated;
    _result.context_full = completion->context_full;
    _result.stopped_eos = completion->stopped_eos;
    _result.stopped_words = completion->stopped_word;
    _result.stopping_word = completion->stopping_word;
    _result.stopped_limited = completion->stopped_limit;

    // Get audio tokens if TTS is enabled
    if (_has_vocoder && _rn_ctx->tts_wrapper != nullptr) {
      _result.audio_tokens = _rn_ctx->tts_wrapper->audio_tokens;
    }
    common_perf_print(_rn_ctx->ctx, _rn_ctx->completion->ctx_sampling);
    // End completion
    completion->endCompletion();

  } catch (const std::exception &e) {
    SetError(e.what());
    return;
  }

  if (_onComplete) {
    _onComplete();
  }
}

void LlamaCompletionWorker::OnOK() {
  auto env = Napi::AsyncWorker::Env();
  const bool vocab_only = _params.vocab_only;
  auto result = Napi::Object::New(env);
  result.Set("chat_format", Napi::Number::New(env, _chat_format));
  result.Set("tokens_evaluated",
             Napi::Number::New(env, _result.tokens_evaluated));
  result.Set("tokens_predicted", Napi::Number::New(Napi::AsyncWorker::Env(),
                                                   _result.tokens_predicted));
  if (_result.draft_tokens > 0 || _result.draft_tokens_accepted > 0) {
    result.Set("draft_tokens",
               Napi::Number::New(env, _result.draft_tokens));
    result.Set("draft_tokens_accepted",
               Napi::Number::New(env, _result.draft_tokens_accepted));
  }
  result.Set("truncated", Napi::Boolean::New(env, _result.truncated));
  result.Set("context_full", Napi::Boolean::New(env, _result.context_full));
  result.Set("interrupted", Napi::Boolean::New(env, _interrupted));
  // A vocab-only completion does not run or rewind the shared completion
  // context, so none of its generated state applies to this result.
  std::string final_text = (!vocab_only && _rn_ctx->completion != nullptr)
                               ? _rn_ctx->completion->generated_text
                               : _result.text;
  result.Set("text", Napi::String::New(env, final_text.c_str()));
  result.Set("stopped_eos", Napi::Boolean::New(env, _result.stopped_eos));
  result.Set("stopped_words", Napi::Boolean::New(env, _result.stopped_words));
  result.Set("stopping_word",
             Napi::String::New(env, _result.stopping_word.c_str()));
  result.Set("stopped_limited",
             Napi::Boolean::New(env, _result.stopped_limited));

  Napi::Array tool_calls = Napi::Array::New(Napi::AsyncWorker::Env());
  std::string reasoning_content = "";
  std::string content;
  if (!vocab_only && !_interrupted && _rn_ctx->completion != nullptr) {
    try {
      auto final_output = _rn_ctx->completion->parseChatOutput(false);
      reasoning_content = final_output.reasoning_content;
      content = final_output.content;

      // Convert tool calls to JavaScript format
      for (size_t i = 0; i < final_output.tool_calls.size(); i++) {
        const auto &tc = final_output.tool_calls[i];
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
    } catch (const std::exception &e) {
      // console_log(env, "Error parsing tool calls: " + std::string(e.what()));
    }
  }
  if (tool_calls.Length() > 0) {
    result.Set("tool_calls", tool_calls);
  }
  if (!reasoning_content.empty()) {
    result.Set("reasoning_content",
               Napi::String::New(env, reasoning_content.c_str()));
  }
  if (!content.empty()) {
    result.Set("content", Napi::String::New(env, content.c_str()));
  }

  // Add audio_tokens if vocoder is enabled and we have audio tokens
  if (_has_vocoder && !_result.audio_tokens.empty()) {
    auto audio_tokens = Napi::Array::New(env, _result.audio_tokens.size());
    for (size_t i = 0; i < _result.audio_tokens.size(); i++) {
      audio_tokens.Set(i, Napi::Number::New(env, _result.audio_tokens[i]));
    }
    result.Set("audio_tokens", audio_tokens);
  }

  // Add completion_probabilities to final result
  if (!vocab_only && _rn_ctx->params.sampling.n_probs > 0 &&
      _rn_ctx->completion != nullptr &&
      !_rn_ctx->completion->generated_token_probs.empty()) {
    result.Set("completion_probabilities", TokenProbsToArray(env, _rn_ctx->ctx, _rn_ctx->completion->generated_token_probs));
  }

  auto timingsResult = Napi::Object::New(Napi::AsyncWorker::Env());
  double prompt_n = 0.0;
  double prompt_ms = 0.0;
  double predicted_n = 0.0;
  double predicted_ms = 0.0;

  // Vocab-only completion exits before inference starts. Reading the shared
  // context here would expose model-load or previous-completion counters,
  // which vary by backend and platform.
  if (!vocab_only) {
    const auto timings_token = llama_perf_context(_rn_ctx->ctx);
    prompt_n = static_cast<double>(timings_token.n_p_eval);
    prompt_ms = timings_token.t_p_eval_ms;
    predicted_n = _rn_ctx->completion != nullptr
                      ? static_cast<double>(
                            _rn_ctx->completion->num_tokens_predicted)
                      : static_cast<double>(_result.tokens_predicted);
    predicted_ms = _rn_ctx->completion != nullptr
                       ? _rn_ctx->completion->t_token_generation * 1e3
                       : timings_token.t_eval_ms;
  }

  timingsResult.Set("prompt_n", Napi::Number::New(Napi::AsyncWorker::Env(),
                                                  prompt_n));
  timingsResult.Set("prompt_ms", Napi::Number::New(Napi::AsyncWorker::Env(),
                                                   prompt_ms));
  const double prompt_per_token_ms =
      prompt_n > 0 ? prompt_ms / prompt_n : 0.0;
  timingsResult.Set("prompt_per_token_ms",
                    Napi::Number::New(Napi::AsyncWorker::Env(),
                                      prompt_per_token_ms));
  const double prompt_per_second =
      prompt_ms > 0 ? 1e3 / prompt_ms * prompt_n : 0.0;
  timingsResult.Set("prompt_per_second",
                    Napi::Number::New(Napi::AsyncWorker::Env(),
                                      prompt_per_second));

  timingsResult.Set("predicted_n", Napi::Number::New(Napi::AsyncWorker::Env(),
                                                     predicted_n));
  timingsResult.Set("predicted_ms", Napi::Number::New(Napi::AsyncWorker::Env(),
                                                      predicted_ms));
  const double predicted_per_token_ms =
      predicted_n > 0 ? predicted_ms / predicted_n : 0.0;
  timingsResult.Set("predicted_per_token_ms",
                    Napi::Number::New(Napi::AsyncWorker::Env(),
                                      predicted_per_token_ms));
  const double predicted_per_second =
      predicted_ms > 0 ? 1e3 / predicted_ms * predicted_n : 0.0;
  timingsResult.Set("predicted_per_second",
                    Napi::Number::New(Napi::AsyncWorker::Env(),
                                      predicted_per_second));

  result.Set("timings", timingsResult);

  Napi::Promise::Deferred::Resolve(result);
}

void LlamaCompletionWorker::OnError(const Napi::Error &err) {
  Napi::Promise::Deferred::Reject(err.Value());
}
