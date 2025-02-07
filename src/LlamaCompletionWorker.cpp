#include "LlamaCompletionWorker.h"
#include "LlamaContext.h"

size_t common_part(const std::vector<llama_token> &a,
                   const std::vector<llama_token> &b) {
  size_t i = 0;
  while (i < a.size() && i < b.size() && a[i] == b[i]) {
    i++;
  }
  return i;
}

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
    Napi::Function callback, common_params params,
    std::vector<std::string> stop_words,
    int32_t chat_format)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _sess(sess),
      _params(params), _stop_words(stop_words), _chat_format(chat_format) {
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

  std::vector<llama_token> prompt_tokens =
      ::common_tokenize(ctx, _params.prompt, add_bos);
  n_input = prompt_tokens.size();
  if (_sess->tokens_ptr()->size() > 0) {
    n_cur = common_part(*(_sess->tokens_ptr()), prompt_tokens);
    if (n_cur == n_input) {
      --n_cur;
    }
    n_input -= n_cur;
    llama_kv_cache_seq_rm(ctx, 0, n_cur, -1);
  }
  _sess->set_tokens(std::move(prompt_tokens));

  const int max_len = _params.n_predict < 0 ? 0 : _params.n_predict;
  _sess->tokens_ptr()->reserve(_sess->tokens_ptr()->size() + max_len);

  auto embd = _sess->tokens_ptr();
  for (int i = 0; i < max_len || _stop; i++) {
    // check if we need to remove some tokens
    if (embd->size() >= _params.n_ctx) {
      const int n_left = n_cur - n_keep - 1;
      const int n_discard = n_left / 2;

      llama_kv_cache_seq_rm(ctx, 0, n_keep + 1, n_keep + n_discard + 1);
      llama_kv_cache_seq_add(ctx, 0, n_keep + 1 + n_discard, n_cur, -n_discard);

      // shift the tokens
      embd->insert(embd->begin() + n_keep + 1,
                   embd->begin() + n_keep + 1 + n_discard, embd->end());
      embd->resize(embd->size() - n_discard);

      n_cur -= n_discard;
      _result.truncated = true;
    }
    int ret = llama_decode(
        ctx, llama_batch_get_one(embd->data() + n_cur, n_input));
    if (ret < 0) {
      SetError("Failed to decode token, code: " + std::to_string(ret));
      break;
    }
    // sample the next token
    const llama_token new_token_id =
        common_sampler_sample(sampling.get(), ctx, -1);
    common_sampler_accept(sampling.get(), new_token_id, true);
    // prepare the next batch
    embd->emplace_back(new_token_id);
    auto token = common_token_to_piece(ctx, new_token_id);
    _result.text += token;
    n_cur += n_input;
    _result.tokens_evaluated += n_input;
    _result.tokens_predicted += 1;
    n_input = 1;
    if (_has_callback) {
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
      break;
    }
    // check for stop words
    if (!_stop_words.empty()) {
      const size_t stop_pos =
          findStoppingStrings(_result.text, token.size(), _stop_words);
      if (stop_pos != std::string::npos) {
        break;
      }
    }
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
  result.Set("tokens_evaluated", Napi::Number::New(env,
                                                   _result.tokens_evaluated));
  result.Set("tokens_predicted", Napi::Number::New(Napi::AsyncWorker::Env(),
                                                   _result.tokens_predicted));
  result.Set("truncated",
             Napi::Boolean::New(env, _result.truncated));
  result.Set("text",
             Napi::String::New(env, _result.text.c_str()));

  Napi::Array tool_calls = Napi::Array::New(Napi::AsyncWorker::Env());
  if (!_stop) {
    try {
      common_chat_msg message = common_chat_parse(_result.text, static_cast<common_chat_format>(_chat_format));
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

  auto ctx = _sess->context();
  const auto timings_token = llama_perf_context(ctx);

  auto timingsResult = Napi::Object::New(Napi::AsyncWorker::Env());
  timingsResult.Set("prompt_n", Napi::Number::New(Napi::AsyncWorker::Env(), timings_token.n_p_eval));
  timingsResult.Set("prompt_ms", Napi::Number::New(Napi::AsyncWorker::Env(), timings_token.t_p_eval_ms));
  timingsResult.Set("prompt_per_token_ms", Napi::Number::New(Napi::AsyncWorker::Env(), timings_token.t_p_eval_ms / timings_token.n_p_eval));
  timingsResult.Set("prompt_per_second", Napi::Number::New(Napi::AsyncWorker::Env(), 1e3 / timings_token.t_p_eval_ms * timings_token.n_p_eval));
  timingsResult.Set("predicted_n", Napi::Number::New(Napi::AsyncWorker::Env(), timings_token.n_eval));
  timingsResult.Set("predicted_ms", Napi::Number::New(Napi::AsyncWorker::Env(), timings_token.t_eval_ms));
  timingsResult.Set("predicted_per_token_ms", Napi::Number::New(Napi::AsyncWorker::Env(), timings_token.t_eval_ms / timings_token.n_eval));
  timingsResult.Set("predicted_per_second", Napi::Number::New(Napi::AsyncWorker::Env(), 1e3 / timings_token.t_eval_ms * timings_token.n_eval));

  result.Set("timings", timingsResult);
  
  Napi::Promise::Deferred::Resolve(result);
}

void LlamaCompletionWorker::OnError(const Napi::Error &err) {
  Napi::Promise::Deferred::Reject(err.Value());
}
