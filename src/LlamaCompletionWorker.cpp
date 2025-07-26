#include "LlamaCompletionWorker.h"
#include "LlamaContext.h"
#include <limits>

size_t findStoppingStrings(const std::string &text,
                           const size_t last_token_size,
                           const std::vector<std::string> &stop_words) {
  size_t stop_pos = std::string::npos;

  for (const std::string &word : stop_words) {
    size_t pos;

    const size_t tmp = word.size() + last_token_size;
    const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

    pos = text.find(word, from_pos);

    if (pos != std::string::npos &&
        (stop_pos == std::string::npos || pos < stop_pos)) {
      stop_pos = pos;
    }
  }

  return stop_pos;
}

LlamaCompletionWorker::LlamaCompletionWorker(
    const Napi::CallbackInfo &info, LlamaSessionPtr &sess,
    Napi::Function callback,
    common_params params,
    std::vector<std::string> stop_words,
    int32_t chat_format,
    bool thinking_forced_open,
    std::string reasoning_format,
    const std::vector<std::string> &media_paths,
    const std::vector<llama_token> &guide_tokens,
    bool has_vocoder,
    tts_type tts_type_val)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _sess(sess),
      _params(params), _stop_words(stop_words), _chat_format(chat_format),
      _thinking_forced_open(thinking_forced_open),
      _reasoning_format(reasoning_format),
      _media_paths(media_paths), _guide_tokens(guide_tokens),
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
  _sess->get_mutex().lock();
  const auto t_main_start = ggml_time_us();
  const size_t n_ctx = _params.n_ctx;
  const auto n_keep = _params.n_keep;
  size_t n_cur = 0;
  size_t n_input = 0;
  const auto model = _sess->model();
  auto vocab = llama_model_get_vocab(model);

  const bool add_bos = llama_vocab_get_add_bos(vocab);
  auto ctx = _sess->context();

  auto sparams = llama_sampler_chain_default_params();

  LlamaCppSampling sampling{common_sampler_init(model, _params.sampling),
                            common_sampler_free};

  // Process media if any are provided
  if (!_media_paths.empty()) {
    auto *mtmd_ctx = _sess->get_mtmd_ctx();

    if (mtmd_ctx != nullptr) {
      // Process the media and get the tokens
      try {
        n_cur = processMediaPrompt(ctx, mtmd_ctx, _sess, _params, _media_paths);
      } catch (const std::exception &e) {
        SetError(e.what());
        _sess->get_mutex().unlock();
        return;
      }

      if (n_cur <= 0) {
        SetError("Failed to process media");
        _sess->get_mutex().unlock();
        return;
      }

      fprintf(stdout,
              "[DEBUG] Media processing successful, n_cur=%zu, tokens=%zu\n",
              n_cur, _sess->tokens_ptr()->size());

      n_input = _sess->tokens_ptr()->size();
      if (n_cur == n_input) {
        --n_cur;
      }
      n_input -= n_cur;
    } else {
      SetError("Multimodal context not initialized");
      _sess->get_mutex().unlock();
      return;
    }
  } else {
    // Text-only path
    std::vector<llama_token> prompt_tokens =
        ::common_tokenize(ctx, _params.prompt, add_bos, true);
    n_input = prompt_tokens.size();

    if (_sess->tokens_ptr()->size() > 0) {
      n_cur = common_tokens_part(*(_sess->tokens_ptr()), prompt_tokens);
      if (n_cur == n_input) {
        --n_cur;
      }
      n_input -= n_cur;
      llama_memory_seq_rm(llama_get_memory(ctx), 0, n_cur, -1);
    }
    // Set the tokens
    _sess->set_tokens(std::move(prompt_tokens));
  }

  const int max_len = _params.n_predict < 0 ? std::numeric_limits<int>::max() : _params.n_predict;
  _sess->tokens_ptr()->reserve(_sess->tokens_ptr()->size() + max_len);

  auto embd = _sess->tokens_ptr();
  for (int i = 0; (i < max_len || _stop) && !_params.vocab_only; i++) {
    // check if we need to remove some tokens
    if (embd->size() >= _params.n_ctx) {
      if (!_params.ctx_shift) {
        // Context is full and ctx_shift is disabled, so we need to stop
        _result.context_full = true;
        break;
      }

      const int n_left = n_cur - n_keep - 1;
      const int n_discard = n_left / 2;

      auto mem = llama_get_memory(ctx);
      llama_memory_seq_rm(mem, 0, n_keep + 1, n_keep + n_discard + 1);
      llama_memory_seq_add(mem, 0, n_keep + 1 + n_discard, n_cur, -n_discard);

      // shift the tokens
      embd->insert(embd->begin() + n_keep + 1,
                   embd->begin() + n_keep + 1 + n_discard, embd->end());
      embd->resize(embd->size() - n_discard);

      n_cur -= n_discard;
      _result.truncated = true;
    }

    // For multimodal input, n_past might already be set
    // Only decode text tokens if we have any input left
    if (n_input > 0) {
      // Decode tokens in batches using n_batch as chunk size
      int n_past_batch = n_cur;
      int n_remaining = n_input;
      
      while (n_remaining > 0) {
        int n_eval = n_remaining;
        if (n_eval > _params.n_batch) {
          n_eval = _params.n_batch;
        }
        
        int ret = llama_decode(ctx, llama_batch_get_one(embd->data() + n_past_batch, n_eval));
        if (ret < 0) {
          SetError("Failed to decode token batch, code: " + std::to_string(ret) + 
                   ", n_eval: " + std::to_string(n_eval) + 
                   ", n_past_batch: " + std::to_string(n_past_batch));
          break;
        }
        
        n_past_batch += n_eval;
        n_remaining -= n_eval;
      }
    }

    // sample the next token
    llama_token new_token_id = common_sampler_sample(sampling.get(), ctx, -1);
    if (_next_token_uses_guide_token && !_guide_tokens.empty() &&
        !llama_vocab_is_control(vocab, new_token_id) &&
        !llama_vocab_is_eog(vocab, new_token_id)) {
      new_token_id = _guide_tokens[0];
      _guide_tokens.erase(_guide_tokens.begin());
    }
    _next_token_uses_guide_token = (new_token_id == 198);
    common_sampler_accept(sampling.get(), new_token_id, true);
    
    // Collect audio tokens for TTS if vocoder is enabled
    if (_has_vocoder) {
      if ((_tts_type == OUTETTS_V0_1 || _tts_type == OUTETTS_V0_2 || _tts_type == OUTETTS_V0_3) && 
          (new_token_id >= 151672 && new_token_id <= 155772)) {
        _result.audio_tokens.push_back(new_token_id);
      }
    }
    
    // prepare the next batch
    embd->emplace_back(new_token_id);
    auto token = common_token_to_piece(ctx, new_token_id);
    _result.text += token;
    n_cur += n_input;
    _result.tokens_evaluated += n_input;
    _result.tokens_predicted += 1;
    n_input = 1;
    if (_has_callback) {
      // TODO: When we got possible stop words (startsWith)
      // we should avoid calling the callback, wait for the next token
      const char *c_token = strdup(token.c_str());
      _tsfn.BlockingCall(c_token, [](Napi::Env env, Napi::Function jsCallback,
                                     const char *value) {
        auto obj = Napi::Object::New(env);
        obj.Set("token", Napi::String::New(env, value));
        delete value;
        jsCallback.Call({obj});
      });
    }
    // is it an end of generation?
    if (llama_vocab_is_eog(vocab, new_token_id)) {
      _result.stopped_eos = true;
      // TODO: EOS token should be cut
      break;
    }
    // check for stop words
    if (!_stop_words.empty()) {
      const size_t stop_pos =
          findStoppingStrings(_result.text, token.size(), _stop_words);
      if (stop_pos != std::string::npos) {
        _result.stopped_words = true;
        _result.stopping_word = _result.text.substr(stop_pos, token.size());
        _result.text = _result.text.substr(0, stop_pos - 1);
        break;
      }
    }
  }
  if (!_result.stopped_eos && !_result.stopped_words) {
    _result.stopped_limited = true;
  }
  const auto t_main_end = ggml_time_us();
  _sess->get_mutex().unlock();
  if (_onComplete) {
    _onComplete();
  }
}

void LlamaCompletionWorker::OnOK() {
  auto env = Napi::AsyncWorker::Env();
  auto result = Napi::Object::New(env);
  result.Set("tokens_evaluated",
             Napi::Number::New(env, _result.tokens_evaluated));
  result.Set("tokens_predicted", Napi::Number::New(Napi::AsyncWorker::Env(),
                                                   _result.tokens_predicted));
  result.Set("truncated", Napi::Boolean::New(env, _result.truncated));
  result.Set("context_full", Napi::Boolean::New(env, _result.context_full));
  result.Set("text", Napi::String::New(env, _result.text.c_str()));
  result.Set("stopped_eos", Napi::Boolean::New(env, _result.stopped_eos));
  result.Set("stopped_words", Napi::Boolean::New(env, _result.stopped_words));
  result.Set("stopping_word",
             Napi::String::New(env, _result.stopping_word.c_str()));
  result.Set("stopped_limited",
             Napi::Boolean::New(env, _result.stopped_limited));

  Napi::Array tool_calls = Napi::Array::New(Napi::AsyncWorker::Env());
  std::string reasoning_content = "";
  std::string content;
  if (!_stop) {
    try {
      common_chat_syntax chat_syntax;
      chat_syntax.format = static_cast<common_chat_format>(_chat_format);
      result.Set("chat_format", Napi::Number::New(env, _chat_format));

      chat_syntax.thinking_forced_open = _thinking_forced_open;

      if (_reasoning_format == "deepseek") {
          chat_syntax.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
      } else if (_reasoning_format == "deepseek-legacy") {
          chat_syntax.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;
      } else {
          chat_syntax.reasoning_format = COMMON_REASONING_FORMAT_NONE;
      }
      common_chat_msg message = common_chat_parse(
          _result.text,
          false,
          chat_syntax
      );
      if (!message.reasoning_content.empty()) {
        reasoning_content = message.reasoning_content;
      }
      if (!message.content.empty()) {
        content = message.content;
      }
      for (size_t i = 0; i < message.tool_calls.size(); i++) {
        const auto &tc = message.tool_calls[i];
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

  auto ctx = _sess->context();
  const auto timings_token = llama_perf_context(ctx);

  auto timingsResult = Napi::Object::New(Napi::AsyncWorker::Env());
  timingsResult.Set("prompt_n", Napi::Number::New(Napi::AsyncWorker::Env(),
                                                  timings_token.n_p_eval));
  timingsResult.Set("prompt_ms", Napi::Number::New(Napi::AsyncWorker::Env(),
                                                   timings_token.t_p_eval_ms));
  timingsResult.Set(
      "prompt_per_token_ms",
      Napi::Number::New(Napi::AsyncWorker::Env(),
                        timings_token.t_p_eval_ms / timings_token.n_p_eval));
  timingsResult.Set("prompt_per_second",
                    Napi::Number::New(Napi::AsyncWorker::Env(),
                                      1e3 / timings_token.t_p_eval_ms *
                                          timings_token.n_p_eval));
  timingsResult.Set("predicted_n", Napi::Number::New(Napi::AsyncWorker::Env(),
                                                     timings_token.n_eval));
  timingsResult.Set("predicted_ms", Napi::Number::New(Napi::AsyncWorker::Env(),
                                                      timings_token.t_eval_ms));
  timingsResult.Set(
      "predicted_per_token_ms",
      Napi::Number::New(Napi::AsyncWorker::Env(),
                        timings_token.t_eval_ms / timings_token.n_eval));
  timingsResult.Set(
      "predicted_per_second",
      Napi::Number::New(Napi::AsyncWorker::Env(),
                        1e3 / timings_token.t_eval_ms * timings_token.n_eval));

  result.Set("timings", timingsResult);

  Napi::Promise::Deferred::Resolve(result);
}

void LlamaCompletionWorker::OnError(const Napi::Error &err) {
  Napi::Promise::Deferred::Reject(err.Value());
}
