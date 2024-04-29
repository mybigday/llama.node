#include "common/common.h"
#include "llama.h"
#include <memory>
#include <mutex>
#include <napi.h>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

typedef std::unique_ptr<llama_model, decltype(&llama_free_model)> LlamaCppModel;
typedef std::unique_ptr<llama_context, decltype(&llama_free)> LlamaCppContext;
typedef std::unique_ptr<llama_sampling_context, decltype(&llama_sampling_free)>
    LlamaCppSampling;
typedef std::unique_ptr<llama_batch, decltype(&llama_batch_free)> LlamaCppBatch;

size_t common_part(const std::vector<llama_token> &a,
                   const std::vector<llama_token> &b) {
  size_t i = 0;
  while (i < a.size() && i < b.size() && a[i] == b[i]) {
    i++;
  }
  return i;
}

template <typename T>
constexpr T get_option(const Napi::Object &options, const std::string &name,
                       const T default_value) {
  if (options.Has(name) && !options.Get(name).IsUndefined() &&
      !options.Get(name).IsNull()) {
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

class LlamaCompletionWorker;

class LlamaContext : public Napi::ObjectWrap<LlamaContext> {
public:
  // construct({ model, embedding, n_ctx, n_batch, n_threads, n_gpu_layers,
  // use_mlock, use_mmap }): LlamaContext throws error
  LlamaContext(const Napi::CallbackInfo &info)
      : Napi::ObjectWrap<LlamaContext>(info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsObject()) {
      Napi::TypeError::New(env, "Object expected").ThrowAsJavaScriptException();
    }
    auto options = info[0].As<Napi::Object>();

    params.model = get_option<std::string>(options, "model", "");
    if (params.model.empty()) {
      Napi::TypeError::New(env, "Model is required")
          .ThrowAsJavaScriptException();
    }
    params.embedding = get_option<bool>(options, "embedding", false);
    params.n_ctx = get_option<int32_t>(options, "n_ctx", 512);
    params.n_batch = get_option<int32_t>(options, "n_batch", 2048);
    params.n_threads =
        get_option<int32_t>(options, "n_threads", get_math_cpu_count() / 2);
    params.n_gpu_layers = get_option<int32_t>(options, "n_gpu_layers", -1);
    params.use_mlock = get_option<bool>(options, "use_mlock", false);
    params.use_mmap = get_option<bool>(options, "use_mmap", true);
    params.numa = static_cast<ggml_numa_strategy>(
        get_option<uint32_t>(options, "numa", 0));

    llama_backend_init();
    llama_numa_init(params.numa);

    auto tuple = llama_init_from_gpt_params(params);
    model.reset(std::get<0>(tuple));
    ctx.reset(std::get<1>(tuple));

    if (model == nullptr || ctx == nullptr) {
      Napi::TypeError::New(env, "Failed to load model")
          .ThrowAsJavaScriptException();
    }
  }

  static void Export(Napi::Env env, Napi::Object &exports) {
    Napi::Function func = DefineClass(
        env, "LlamaContext",
        {InstanceMethod<&LlamaContext::GetSystemInfo>(
             "getSystemInfo",
             static_cast<napi_property_attributes>(napi_enumerable)),
         InstanceMethod<&LlamaContext::Completion>(
             "completion",
             static_cast<napi_property_attributes>(napi_enumerable)),
         InstanceMethod<&LlamaContext::StopCompletion>(
             "stopCompletion",
             static_cast<napi_property_attributes>(napi_enumerable)),
         InstanceMethod<&LlamaContext::SaveSession>(
             "saveSession",
             static_cast<napi_property_attributes>(napi_enumerable)),
         InstanceMethod<&LlamaContext::LoadSession>(
             "loadSession",
             static_cast<napi_property_attributes>(napi_enumerable)),
         InstanceMethod<&LlamaContext::Release>(
             "release",
             static_cast<napi_property_attributes>(napi_enumerable))});
    Napi::FunctionReference *constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
#if NAPI_VERSION > 5
    env.SetInstanceData(constructor);
#endif
    exports.Set("LlamaContext", func);
  }

  llama_context *getContext() { return ctx.get(); }
  llama_model *getModel() { return model.get(); }

  std::vector<llama_token> *getTokens() { return tokens.get(); }

  const gpt_params &getParams() const { return params; }

  void ensureTokens() {
    if (tokens == nullptr) {
      tokens = std::make_unique<std::vector<llama_token>>();
    }
  }

  void setTokens(std::vector<llama_token> tokens) {
    this->tokens.reset(new std::vector<llama_token>(std::move(tokens)));
  }

  std::mutex &getMutex() { return mutex; }

  void Dispose() {
    std::lock_guard<std::mutex> lock(mutex);
    compl_worker = nullptr;
    ctx.reset();
    tokens.reset();
    model.reset();
  }

private:
  Napi::Value GetSystemInfo(const Napi::CallbackInfo &info);
  Napi::Value Completion(const Napi::CallbackInfo &info);
  void StopCompletion(const Napi::CallbackInfo &info);
  Napi::Value SaveSession(const Napi::CallbackInfo &info);
  Napi::Value LoadSession(const Napi::CallbackInfo &info);
  Napi::Value Release(const Napi::CallbackInfo &info);

  gpt_params params;
  LlamaCppModel model{nullptr, llama_free_model};
  LlamaCppContext ctx{nullptr, llama_free};
  std::unique_ptr<std::vector<llama_token>> tokens;
  std::mutex mutex;
  LlamaCompletionWorker *compl_worker = nullptr;
};

class LlamaCompletionWorker : public Napi::AsyncWorker,
                              public Napi::Promise::Deferred {
  LlamaContext *_ctx;
  gpt_params _params;
  std::vector<std::string> _stop_words;
  std::string generated_text = "";
  Napi::ThreadSafeFunction _tsfn;
  bool _has_callback = false;
  bool _stop = false;
  size_t tokens_predicted = 0;
  size_t tokens_evaluated = 0;
  bool truncated = false;

public:
  LlamaCompletionWorker(const Napi::CallbackInfo &info, LlamaContext *ctx,
                        Napi::Function callback, gpt_params params,
                        std::vector<std::string> stop_words = {})
      : AsyncWorker(info.Env()), Deferred(info.Env()), _ctx(ctx),
        _params(params), _stop_words(stop_words) {
    _ctx->Ref();
    if (!callback.IsEmpty()) {
      _tsfn = Napi::ThreadSafeFunction::New(info.Env(), callback,
                                            "LlamaCompletionCallback", 0, 1);
      _has_callback = true;
    }
  }

  ~LlamaCompletionWorker() {
    _ctx->Unref();
    if (_has_callback) {
      _tsfn.Abort();
      _tsfn.Release();
    }
  }

  void Stop() { _stop = true; }

protected:
  size_t findStoppingStrings(const std::string &text,
                             const size_t last_token_size) {
    size_t stop_pos = std::string::npos;

    for (const std::string &word : _stop_words) {
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

  void Execute() {
    _ctx->getMutex().lock();
    _ctx->ensureTokens();
    const auto t_main_start = ggml_time_us();
    const size_t n_ctx = _params.n_ctx;
    auto n_keep = _params.n_keep;
    auto n_predict = _params.n_predict;
    size_t n_cur = 0;
    size_t n_input = 0;
    const bool add_bos = llama_should_add_bos_token(_ctx->getModel());
    auto *ctx = _ctx->getContext();

    llama_set_rng_seed(ctx, _params.seed);

    LlamaCppSampling sampling{llama_sampling_init(_params.sparams),
                              llama_sampling_free};

    std::vector<llama_token> prompt_tokens =
        ::llama_tokenize(ctx, _params.prompt, add_bos);
    n_input = prompt_tokens.size();
    if (_ctx->getTokens() != nullptr) {
      n_cur = common_part(*_ctx->getTokens(), prompt_tokens);
      if (n_cur == n_input) {
        --n_cur;
      }
      n_input -= n_cur;
      llama_kv_cache_seq_rm(ctx, 0, n_cur, -1);
    }
    _ctx->setTokens(std::move(prompt_tokens));

    const int max_len = _params.n_predict < 0 ? 0 : _params.n_predict;

    for (int i = 0; i < max_len || _stop; i++) {
      auto *embd = _ctx->getTokens();
      // check if we need to remove some tokens
      if (embd->size() >= n_ctx) {
        const int n_left = n_cur - n_keep - 1;
        const int n_discard = n_left / 2;

        llama_kv_cache_seq_rm(ctx, 0, n_keep + 1, n_keep + n_discard + 1);
        llama_kv_cache_seq_add(ctx, 0, n_keep + 1 + n_discard, n_cur,
                               -n_discard);

        for (size_t i = n_keep + 1 + n_discard; i < embd->size(); i++) {
          (*embd)[i - n_discard] = (*embd)[i];
        }
        embd->resize(embd->size() - n_discard);

        n_cur -= n_discard;
        truncated = true;
      }
      int ret = llama_decode(
          ctx, llama_batch_get_one(embd->data() + n_cur, n_input, n_cur, 0));
      if (ret < 0) {
        SetError("Failed to decode token, code: " + std::to_string(ret));
        break;
      }
      // sample the next token
      const llama_token new_token_id =
          llama_sampling_sample(sampling.get(), ctx, nullptr);
      // prepare the next batch
      embd->push_back(new_token_id);
      auto token = llama_token_to_piece(ctx, new_token_id);
      generated_text += token;
      n_cur += n_input;
      tokens_evaluated += n_input;
      tokens_predicted += 1;
      n_input = 1;
      if (_has_callback) {
        const char *c_token = strdup(token.c_str());
        _tsfn.BlockingCall(c_token, [](Napi::Env env, Napi::Function jsCallback,
                                       const char *value) {
          auto obj = Napi::Object::New(env);
          obj.Set("token", Napi::String::New(env, value));
          jsCallback.Call({obj});
        });
      }
      // is it an end of generation?
      if (llama_token_is_eog(_ctx->getModel(), new_token_id)) {
        break;
      }
      // check for stop words
      if (!_stop_words.empty()) {
        const size_t stop_pos =
            findStoppingStrings(generated_text, token.size());
        if (stop_pos != std::string::npos) {
          break;
        }
      }
    }
    const auto t_main_end = ggml_time_us();
    _ctx->getMutex().unlock();
  }

  void OnOK() {
    auto result = Napi::Object::New(Napi::AsyncWorker::Env());
    result.Set("tokens_evaluated",
               Napi::Number::New(Napi::AsyncWorker::Env(), tokens_evaluated));
    result.Set("tokens_predicted",
               Napi::Number::New(Napi::AsyncWorker::Env(), tokens_predicted));
    result.Set("truncated",
               Napi::Boolean::New(Napi::AsyncWorker::Env(), truncated));
    result.Set("text",
               Napi::String::New(Napi::AsyncWorker::Env(), generated_text));
    Napi::Promise::Deferred::Resolve(result);
  }

  void OnError(const Napi::Error &err) {
    Napi::Promise::Deferred::Reject(err.Value());
  }
};

class SaveSessionWorker : public Napi::AsyncWorker,
                          public Napi::Promise::Deferred {
  std::string _path;
  LlamaContext *_ctx;

public:
  SaveSessionWorker(const Napi::CallbackInfo &info, LlamaContext *ctx)
      : AsyncWorker(info.Env()), Deferred(info.Env()),
        _path(info[0].ToString()), _ctx(ctx) {
    _ctx->Ref();
  }

protected:
  void Execute() {
    _ctx->getMutex().lock();
    if (_ctx->getTokens() == nullptr) {
      SetError("Failed to save session");
      return;
    }
    if (!llama_state_save_file(_ctx->getContext(), _path.c_str(),
                               _ctx->getTokens()->data(),
                               _ctx->getTokens()->size())) {
      SetError("Failed to save session");
    }
    _ctx->getMutex().unlock();
  }

  void OnOK() { Resolve(AsyncWorker::Env().Undefined()); }

  void OnError(const Napi::Error &err) { Reject(err.Value()); }
};

class LoadSessionWorker : public Napi::AsyncWorker,
                          public Napi::Promise::Deferred {
  std::string _path;
  LlamaContext *_ctx;
  size_t count = 0;

public:
  LoadSessionWorker(const Napi::CallbackInfo &info, LlamaContext *ctx)
      : AsyncWorker(info.Env()), Deferred(info.Env()),
        _path(info[0].ToString()), _ctx(ctx) {
    _ctx->Ref();
  }

protected:
  void Execute() {
    _ctx->getMutex().lock();
    _ctx->ensureTokens();
    // reserve the maximum number of tokens for capacity
    _ctx->getTokens()->reserve(_ctx->getParams().n_ctx);
    if (!llama_state_load_file(_ctx->getContext(), _path.c_str(),
                               _ctx->getTokens()->data(),
                               _ctx->getTokens()->capacity(), &count)) {
      SetError("Failed to load session");
    }
    _ctx->getMutex().unlock();
  }

  void OnOK() { Resolve(AsyncWorker::Env().Undefined()); }

  void OnError(const Napi::Error &err) { Reject(err.Value()); }
};

class DisposeWorker : public Napi::AsyncWorker, public Napi::Promise::Deferred {
public:
  DisposeWorker(Napi::Env env, LlamaContext *ctx)
      : AsyncWorker(env), Deferred(env), ctx_(ctx) {
    ctx_->Ref();
  }

  ~DisposeWorker() { ctx_->Unref(); }

protected:
  void Execute() override { ctx_->Dispose(); }

  void OnOK() override { Resolve(AsyncWorker::Env().Undefined()); }

  void OnError(const Napi::Error &err) override { Reject(err.Value()); }

private:
  LlamaContext *ctx_;
};

// getSystemInfo(): string
Napi::Value LlamaContext::GetSystemInfo(const Napi::CallbackInfo &info) {
  return Napi::String::New(info.Env(), get_system_info(params).c_str());
}

// completion(options: LlamaCompletionOptions, onToken?: (token: string) =>
// void): Promise<LlamaCompletionResult>
Napi::Value LlamaContext::Completion(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsObject()) {
    Napi::TypeError::New(env, "Object expected").ThrowAsJavaScriptException();
  }
  if (info.Length() >= 2 && !info[1].IsFunction()) {
    Napi::TypeError::New(env, "Function expected").ThrowAsJavaScriptException();
  }
  auto options = info[0].As<Napi::Object>();

  gpt_params params;
  params.prompt = get_option<std::string>(options, "prompt", "");
  if (params.prompt.empty()) {
    Napi::TypeError::New(env, "Prompt is required")
        .ThrowAsJavaScriptException();
  }
  params.n_predict = get_option<int32_t>(options, "n_predict", -1);
  params.sparams.temp = get_option<float>(options, "temperature", 0.80f);
  params.sparams.top_k = get_option<int32_t>(options, "top_k", 40);
  params.sparams.top_p = get_option<float>(options, "top_p", 0.95f);
  params.sparams.min_p = get_option<float>(options, "min_p", 0.05f);
  params.sparams.tfs_z = get_option<float>(options, "tfs_z", 1.00f);
  params.sparams.mirostat = get_option<int32_t>(options, "mirostat", 0.00f);
  params.sparams.mirostat_tau =
      get_option<float>(options, "mirostat_tau", 5.00f);
  params.sparams.mirostat_eta =
      get_option<float>(options, "mirostat_eta", 0.10f);
  params.sparams.penalty_last_n =
      get_option<int32_t>(options, "penalty_last_n", 64);
  params.sparams.penalty_repeat =
      get_option<float>(options, "penalty_repeat", 1.00f);
  params.sparams.penalty_freq =
      get_option<float>(options, "penalty_freq", 0.00f);
  params.sparams.penalty_present =
      get_option<float>(options, "penalty_present", 0.00f);
  params.sparams.penalize_nl = get_option<bool>(options, "penalize_nl", false);
  params.sparams.typical_p = get_option<float>(options, "typical_p", 1.00f);
  params.ignore_eos = get_option<float>(options, "ignore_eos", false);
  params.sparams.grammar = get_option<std::string>(options, "grammar", "");
  params.n_keep = get_option<int32_t>(options, "n_keep", 0);
  params.seed = get_option<int32_t>(options, "seed", LLAMA_DEFAULT_SEED);
  std::vector<std::string> stop_words;
  if (options.Has("stop") && options.Get("stop").IsArray()) {
    auto stop_words_array = options.Get("stop").As<Napi::Array>();
    for (size_t i = 0; i < stop_words_array.Length(); i++) {
      stop_words.push_back(stop_words_array.Get(i).ToString().Utf8Value());
    }
  }

  // options.on_sample
  Napi::Function callback;
  if (info.Length() >= 2) {
    callback = info[1].As<Napi::Function>();
  }

  auto worker =
      new LlamaCompletionWorker(info, this, callback, params, stop_words);
  worker->Queue();
  compl_worker = worker;
  return worker->Promise();
}

// stopCompletion(): void
void LlamaContext::StopCompletion(const Napi::CallbackInfo &info) {
  if (compl_worker != nullptr) {
    compl_worker->Stop();
  }
}

// saveSession(path: string): Promise<void> throws error
Napi::Value LlamaContext::SaveSession(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(env, "String expected").ThrowAsJavaScriptException();
  }
  auto *worker = new SaveSessionWorker(info, this);
  worker->Queue();
  return worker->Promise();
}

// loadSession(path: string): Promise<{ count }> throws error
Napi::Value LlamaContext::LoadSession(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(env, "String expected").ThrowAsJavaScriptException();
  }
  auto *worker = new LoadSessionWorker(info, this);
  worker->Queue();
  return worker->Promise();
}

// release(): Promise<void>
Napi::Value LlamaContext::Release(const Napi::CallbackInfo &info) {
  if (compl_worker != nullptr) {
    compl_worker->Stop();
  }
  auto *worker = new DisposeWorker(info.Env(), this);
  worker->Queue();
  return worker->Promise();
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  LlamaContext::Export(env, exports);
  return exports;
}

NODE_API_MODULE(addons, Init)
