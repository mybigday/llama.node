#include "ggml.h"
#include "gguf.h"
#include "llama-impl.h"
#include "LlamaContext.h"
#include "DetokenizeWorker.h"
#include "DisposeWorker.h"
#include "EmbeddingWorker.h"
#include "LlamaCompletionWorker.h"
#include "LoadSessionWorker.h"
#include "SaveSessionWorker.h"
#include "TokenizeWorker.h"

Napi::Object LlamaContext::ModelInfo(Napi::Env env, const std::string & path) {
  struct gguf_init_params params = {
    /*.no_alloc = */ false,
    /*.ctx      = */ NULL,
  };
  struct gguf_context * ctx = gguf_init_from_file(path.c_str(), params);

  Napi::Object info = Napi::Object::New(env);
  info.Set("version", Napi::Number::New(env, gguf_get_version(ctx)));
  info.Set("alignment", Napi::Number::New(env, gguf_get_alignment(ctx)));
  info.Set("data_offset", Napi::Number::New(env, gguf_get_data_offset(ctx)));

  // kv
  {
    const int n_kv = gguf_get_n_kv(ctx);

    for (int i = 0; i < n_kv; ++i) {
      const char * key = gguf_get_key(ctx, i);
      const std::string value = gguf_kv_to_str(ctx, i);
      info.Set(key, Napi::String::New(env, value.c_str()));
    }
  }

  gguf_free(ctx);

  return info;
}

std::vector<common_chat_msg> get_messages(Napi::Array messages) {
  std::vector<common_chat_msg> chat;
  for (size_t i = 0; i < messages.Length(); i++) {
    auto message = messages.Get(i).As<Napi::Object>();
    chat.push_back({
      get_option<std::string>(message, "role", ""),
      get_option<std::string>(message, "content", ""),
    });
  }
  return std::move(chat);
}

void LlamaContext::Init(Napi::Env env, Napi::Object &exports) {
  Napi::Function func = DefineClass(
      env, "LlamaContext",
      {InstanceMethod<&LlamaContext::GetSystemInfo>(
           "getSystemInfo",
           static_cast<napi_property_attributes>(napi_enumerable)),
       InstanceMethod<&LlamaContext::GetModelInfo>(
           "getModelInfo",
           static_cast<napi_property_attributes>(napi_enumerable)),
       InstanceMethod<&LlamaContext::GetFormattedChat>(
           "getFormattedChat",
           static_cast<napi_property_attributes>(napi_enumerable)),
       InstanceMethod<&LlamaContext::Completion>(
           "completion",
           static_cast<napi_property_attributes>(napi_enumerable)),
       InstanceMethod<&LlamaContext::StopCompletion>(
           "stopCompletion",
           static_cast<napi_property_attributes>(napi_enumerable)),
       InstanceMethod<&LlamaContext::Tokenize>(
           "tokenize", static_cast<napi_property_attributes>(napi_enumerable)),
       InstanceMethod<&LlamaContext::Detokenize>(
           "detokenize",
           static_cast<napi_property_attributes>(napi_enumerable)),
       InstanceMethod<&LlamaContext::Embedding>(
           "embedding", static_cast<napi_property_attributes>(napi_enumerable)),
       InstanceMethod<&LlamaContext::SaveSession>(
           "saveSession",
           static_cast<napi_property_attributes>(napi_enumerable)),
       InstanceMethod<&LlamaContext::LoadSession>(
           "loadSession",
           static_cast<napi_property_attributes>(napi_enumerable)),
       InstanceMethod<&LlamaContext::Release>(
           "release", static_cast<napi_property_attributes>(napi_enumerable))});
  Napi::FunctionReference *constructor = new Napi::FunctionReference();
  *constructor = Napi::Persistent(func);
#if NAPI_VERSION > 5
  env.SetInstanceData(constructor);
#endif
  exports.Set("LlamaContext", func);
}

const std::vector<ggml_type> kv_cache_types = {
  GGML_TYPE_F32,
  GGML_TYPE_F16,
  GGML_TYPE_BF16,
  GGML_TYPE_Q8_0,
  GGML_TYPE_Q4_0,
  GGML_TYPE_Q4_1,
  GGML_TYPE_IQ4_NL,
  GGML_TYPE_Q5_0,
  GGML_TYPE_Q5_1,
};

static ggml_type kv_cache_type_from_str(const std::string & s) {
  for (const auto & type : kv_cache_types) {
    if (ggml_type_name(type) == s) {
      return type;
    }
  }
  throw std::runtime_error("Unsupported cache type: " + s);
}

static int32_t pooling_type_from_str(const std::string & s) {
  if (s == "none") return LLAMA_POOLING_TYPE_NONE;
  if (s == "mean") return LLAMA_POOLING_TYPE_MEAN;
  if (s == "cls") return LLAMA_POOLING_TYPE_CLS;
  if (s == "last") return LLAMA_POOLING_TYPE_LAST;
  if (s == "rank") return LLAMA_POOLING_TYPE_RANK;
  return LLAMA_POOLING_TYPE_UNSPECIFIED;
}

// construct({ model, embedding, n_ctx, n_batch, n_threads, n_gpu_layers,
// use_mlock, use_mmap }): LlamaContext throws error
LlamaContext::LlamaContext(const Napi::CallbackInfo &info)
    : Napi::ObjectWrap<LlamaContext>(info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsObject()) {
    Napi::TypeError::New(env, "Object expected").ThrowAsJavaScriptException();
  }
  auto options = info[0].As<Napi::Object>();

  common_params params;
  params.model = get_option<std::string>(options, "model", "");
  if (params.model.empty()) {
    Napi::TypeError::New(env, "Model is required").ThrowAsJavaScriptException();
  }

  params.vocab_only = get_option<bool>(options, "vocab_only", false);
  if (params.vocab_only) {
    params.warmup = false;
  }

  params.n_ctx = get_option<int32_t>(options, "n_ctx", 512);
  params.n_batch = get_option<int32_t>(options, "n_batch", 2048);
  params.n_ubatch = get_option<int32_t>(options, "n_ubatch", 512);
  params.embedding = get_option<bool>(options, "embedding", false);
  if (params.embedding) {
    // For non-causal models, batch size must be equal to ubatch size
    params.n_ubatch = params.n_batch;
  }
  params.embd_normalize = get_option<int32_t>(options, "embd_normalize", 2);
  params.pooling_type = (enum llama_pooling_type) pooling_type_from_str(
    get_option<std::string>(options, "pooling_type", "").c_str()
  );

  params.cpuparams.n_threads =
      get_option<int32_t>(options, "n_threads", cpu_get_num_math() / 2);
  params.n_gpu_layers = get_option<int32_t>(options, "n_gpu_layers", -1);
  params.flash_attn = get_option<bool>(options, "flash_attn", false);
  params.cache_type_k = kv_cache_type_from_str(get_option<std::string>(options, "cache_type_k", "f16").c_str());
  params.cache_type_v = kv_cache_type_from_str(get_option<std::string>(options, "cache_type_v", "f16").c_str());

  params.use_mlock = get_option<bool>(options, "use_mlock", false);
  params.use_mmap = get_option<bool>(options, "use_mmap", true);
  params.numa =
      static_cast<ggml_numa_strategy>(get_option<uint32_t>(options, "numa", 0));

  llama_backend_init();
  llama_numa_init(params.numa);

  auto sess = std::make_shared<LlamaSession>(params);

  if (sess->model() == nullptr || sess->context() == nullptr) {
    Napi::TypeError::New(env, "Failed to load model")
        .ThrowAsJavaScriptException();
  }

  _sess = sess;
  _info = common_params_get_system_info(params);
}

// getSystemInfo(): string
Napi::Value LlamaContext::GetSystemInfo(const Napi::CallbackInfo &info) {
  return Napi::String::New(info.Env(), _info);
}

bool validateModelChatTemplate(const struct llama_model * model) {
    std::vector<char> model_template(2048, 0); // longest known template is about 1200 bytes
    std::string template_key = "tokenizer.chat_template";
    int32_t res = llama_model_meta_val_str(model, template_key.c_str(), model_template.data(), model_template.size());
    if (res >= 0) {
        llama_chat_message chat[] = {{"user", "test"}};
        const char * tmpl = llama_model_chat_template(model);
        int32_t chat_res = llama_chat_apply_template(tmpl, chat, 1, true, nullptr, 0);
        return chat_res > 0;
    }
    return res > 0;
}

// getModelInfo(): object
Napi::Value LlamaContext::GetModelInfo(const Napi::CallbackInfo &info) {
  char desc[1024];
  auto model = _sess->model();
  llama_model_desc(model, desc, sizeof(desc));

  int count = llama_model_meta_count(model);
  Napi::Object metadata = Napi::Object::New(info.Env());
  for (int i = 0; i < count; i++) {
    char key[256];
    llama_model_meta_key_by_index(model, i, key, sizeof(key));
    char val[2048];
    llama_model_meta_val_str_by_index(model, i, val, sizeof(val));

    metadata.Set(key, val);
  }
  Napi::Object details = Napi::Object::New(info.Env());
  details.Set("desc", desc);
  details.Set("nParams", llama_model_n_params(model));
  details.Set("size", llama_model_size(model));
  details.Set("isChatTemplateSupported", validateModelChatTemplate(model));
  details.Set("metadata", metadata);
  return details;
}

// getFormattedChat(messages: [{ role: string, content: string }]): string
Napi::Value LlamaContext::GetFormattedChat(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsArray()) {
    Napi::TypeError::New(env, "Array expected").ThrowAsJavaScriptException();
  }
  auto messages = info[0].As<Napi::Array>();
  auto formatted = common_chat_apply_template(_sess->model(), "", get_messages(messages), true);
  return Napi::String::New(env, formatted);
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
  if (_sess == nullptr) {
    Napi::TypeError::New(env, "Context is disposed")
        .ThrowAsJavaScriptException();
  }
  if (_wip != nullptr) {
    Napi::TypeError::New(env, "Another completion is in progress")
        .ThrowAsJavaScriptException();
  }
  auto options = info[0].As<Napi::Object>();

  common_params params = _sess->params();
  if (options.Has("messages") && options.Get("messages").IsArray()) {
    auto messages = options.Get("messages").As<Napi::Array>();
    auto formatted = common_chat_apply_template(_sess->model(), "", get_messages(messages), true);
    params.prompt = formatted;
  } else {
    params.prompt = get_option<std::string>(options, "prompt", "");
  }
  if (params.prompt.empty()) {
    Napi::TypeError::New(env, "Prompt is required")
        .ThrowAsJavaScriptException();
  }
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
  params.sampling.xtc_threshold = get_option<float>(options, "xtc_threshold", 0.00f);
  params.sampling.xtc_probability = get_option<float>(options, "xtc_probability", 0.10f);
  params.sampling.dry_multiplier = get_option<float>(options, "dry_multiplier", 1.75f);
  params.sampling.dry_base = get_option<float>(options, "dry_base", 2);
  params.sampling.dry_allowed_length = get_option<float>(options, "dry_allowed_length", -1);
  params.sampling.dry_penalty_last_n = get_option<float>(options, "dry_penalty_last_n", 0);
  params.sampling.ignore_eos = get_option<bool>(options, "ignore_eos", false);
  params.sampling.grammar = get_option<std::string>(options, "grammar", "");
  params.n_keep = get_option<int32_t>(options, "n_keep", 0);
  params.sampling.seed = get_option<int32_t>(options, "seed", LLAMA_DEFAULT_SEED);
  std::vector<std::string> stop_words;
  if (options.Has("stop") && options.Get("stop").IsArray()) {
    auto stop_words_array = options.Get("stop").As<Napi::Array>();
    for (size_t i = 0; i < stop_words_array.Length(); i++) {
      stop_words.push_back(stop_words_array.Get(i).ToString().Utf8Value());
    }
  }

  Napi::Function callback;
  if (info.Length() >= 2) {
    callback = info[1].As<Napi::Function>();
  }

  auto *worker =
      new LlamaCompletionWorker(info, _sess, callback, params, stop_words);
  worker->Queue();
  _wip = worker;
  worker->onComplete([this]() { _wip = nullptr; });
  return worker->Promise();
}

// stopCompletion(): void
void LlamaContext::StopCompletion(const Napi::CallbackInfo &info) {
  if (_wip != nullptr) {
    _wip->Stop();
  }
}

// tokenize(text: string): Promise<TokenizeResult>
Napi::Value LlamaContext::Tokenize(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(env, "String expected").ThrowAsJavaScriptException();
  }
  if (_sess == nullptr) {
    Napi::TypeError::New(env, "Context is disposed")
        .ThrowAsJavaScriptException();
  }
  auto text = info[0].ToString().Utf8Value();
  auto *worker = new TokenizeWorker(info, _sess, text);
  worker->Queue();
  return worker->Promise();
}

// detokenize(tokens: number[]): Promise<string>
Napi::Value LlamaContext::Detokenize(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsArray()) {
    Napi::TypeError::New(env, "Array expected").ThrowAsJavaScriptException();
  }
  if (_sess == nullptr) {
    Napi::TypeError::New(env, "Context is disposed")
        .ThrowAsJavaScriptException();
  }
  auto tokens = info[0].As<Napi::Array>();
  std::vector<int32_t> token_ids;
  for (size_t i = 0; i < tokens.Length(); i++) {
    token_ids.push_back(tokens.Get(i).ToNumber().Int32Value());
  }
  auto *worker = new DetokenizeWorker(info, _sess, token_ids);
  worker->Queue();
  return worker->Promise();
}

// embedding(text: string): Promise<EmbeddingResult>
Napi::Value LlamaContext::Embedding(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(env, "String expected").ThrowAsJavaScriptException();
  }
  if (_sess == nullptr) {
    Napi::TypeError::New(env, "Context is disposed")
        .ThrowAsJavaScriptException();
  }
  auto options = Napi::Object::New(env);
  if (info.Length() >= 2 && info[1].IsObject()) {
    options = info[1].As<Napi::Object>();
  }

  common_params embdParams;
  embdParams.embedding = true;
  embdParams.embd_normalize = get_option<int32_t>(options, "embd_normalize", 2);
  auto text = info[0].ToString().Utf8Value();
  auto *worker = new EmbeddingWorker(info, _sess, text, embdParams);
  worker->Queue();
  return worker->Promise();
}

// saveSession(path: string): Promise<void> throws error
Napi::Value LlamaContext::SaveSession(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(env, "String expected").ThrowAsJavaScriptException();
  }
  if (_sess == nullptr) {
    Napi::TypeError::New(env, "Context is disposed")
        .ThrowAsJavaScriptException();
  }
#ifdef GGML_USE_VULKAN
  if (_sess->params().n_gpu_layers > 0) {
    Napi::TypeError::New(env, "Vulkan cannot save session")
        .ThrowAsJavaScriptException();
  }
#endif
  auto *worker = new SaveSessionWorker(info, _sess);
  worker->Queue();
  return worker->Promise();
}

// loadSession(path: string): Promise<{ count }> throws error
Napi::Value LlamaContext::LoadSession(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(env, "String expected").ThrowAsJavaScriptException();
  }
  if (_sess == nullptr) {
    Napi::TypeError::New(env, "Context is disposed")
        .ThrowAsJavaScriptException();
  }
#ifdef GGML_USE_VULKAN
  if (_sess->params().n_gpu_layers > 0) {
    Napi::TypeError::New(env, "Vulkan cannot load session")
        .ThrowAsJavaScriptException();
  }
#endif
  auto *worker = new LoadSessionWorker(info, _sess);
  worker->Queue();
  return worker->Promise();
}

// release(): Promise<void>
Napi::Value LlamaContext::Release(const Napi::CallbackInfo &info) {
  auto env = info.Env();
  if (_wip != nullptr) {
    _wip->Stop();
  }
  if (_sess == nullptr) {
    auto promise = Napi::Promise::Deferred(env);
    promise.Resolve(env.Undefined());
    return promise.Promise();
  }
  auto *worker = new DisposeWorker(info, std::move(_sess));
  worker->Queue();
  return worker->Promise();
}
