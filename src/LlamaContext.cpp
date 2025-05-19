#include "ggml.h"
#include "gguf.h"
#include "llama-impl.h"
#include "json.hpp"
#include "json-schema-to-grammar.h"
#include "LlamaContext.h"
#include "DetokenizeWorker.h"
#include "DisposeWorker.h"
#include "EmbeddingWorker.h"
#include "LlamaCompletionWorker.h"
#include "LoadSessionWorker.h"
#include "SaveSessionWorker.h"
#include "TokenizeWorker.h"

// Helper function for formatted strings (for console logs)
template<typename ... Args>
static std::string format_string(const std::string& format, Args ... args) {
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; // +1 for null terminator
    if (size_s <= 0) { return "Error formatting string"; }
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1); // -1 to exclude null terminator
}

// Computes FNV-1a hash of the data
static std::string fnv_hash(const uint8_t* data, size_t len) {
  const uint64_t fnv_prime = 0x100000001b3ULL;
  uint64_t hash = 0xcbf29ce484222325ULL;

  for (size_t i = 0; i < len; ++i) {
    hash ^= data[i];
    hash *= fnv_prime;
  }
  return std::to_string(hash);
}

static const std::string base64_chars =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  "abcdefghijklmnopqrstuvwxyz"
  "0123456789+/";

// Base64 decoding function
static std::vector<uint8_t> base64_decode(const std::string &encoded_string) {
  std::vector<uint8_t> decoded;
  int in_len = encoded_string.size();
  int i = 0;
  int j = 0;
  int in_ = 0;
  unsigned char char_array_4[4], char_array_3[3];

  while (in_len-- && (encoded_string[in_] != '=')) {
    if (isspace(encoded_string[in_])) {
      in_++;
      continue;
    }

    if (encoded_string[in_] == '=' || base64_chars.find(encoded_string[in_]) == std::string::npos) {
      break;
    }

    char_array_4[i++] = encoded_string[in_]; in_++;
    if (i == 4) {
      for (i = 0; i < 4; i++) {
        char_array_4[i] = base64_chars.find(char_array_4[i]);
      }

      char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
      char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

      for (i = 0; i < 3; i++) {
        decoded.push_back(char_array_3[i]);
      }
      i = 0;
    }
  }

  if (i) {
    for (j = i; j < 4; j++) {
      char_array_4[j] = 0;
    }

    for (j = 0; j < 4; j++) {
      char_array_4[j] = base64_chars.find(char_array_4[j]);
    }

    char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
    char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
    char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

    for (j = 0; j < i - 1; j++) {
      decoded.push_back(char_array_3[j]);
    }
  }

  return decoded;
}

using json = nlohmann::ordered_json;

// loadModelInfo(path: string): object
Napi::Value LlamaContext::ModelInfo(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  struct gguf_init_params params = {
    /*.no_alloc = */ false,
    /*.ctx      = */ NULL,
  };
  std::string path = info[0].ToString().Utf8Value();
  
  // Convert Napi::Array to vector<string>
  std::vector<std::string> skip;
  if (info.Length() > 1 && info[1].IsArray()) {
    Napi::Array skipArray = info[1].As<Napi::Array>();
    for (uint32_t i = 0; i < skipArray.Length(); i++) {
      skip.push_back(skipArray.Get(i).ToString().Utf8Value());
    }
  }

  struct gguf_context * ctx = gguf_init_from_file(path.c_str(), params);

  Napi::Object metadata = Napi::Object::New(env);
  if (std::find(skip.begin(), skip.end(), "version") == skip.end()) {
    metadata.Set("version", Napi::Number::New(env, gguf_get_version(ctx)));
  }
  if (std::find(skip.begin(), skip.end(), "alignment") == skip.end()) {
    metadata.Set("alignment", Napi::Number::New(env, gguf_get_alignment(ctx)));
  }
  if (std::find(skip.begin(), skip.end(), "data_offset") == skip.end()) {
    metadata.Set("data_offset", Napi::Number::New(env, gguf_get_data_offset(ctx)));
  }

  // kv
  {
    const int n_kv = gguf_get_n_kv(ctx);

    for (int i = 0; i < n_kv; ++i) {
      const char * key = gguf_get_key(ctx, i);
      if (std::find(skip.begin(), skip.end(), key) != skip.end()) {
        continue;
      }
      const std::string value = gguf_kv_to_str(ctx, i);
      metadata.Set(key, Napi::String::New(env, value.c_str()));
    }
  }

  gguf_free(ctx);

  return metadata;
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
       InstanceMethod<&LlamaContext::ApplyLoraAdapters>(
           "applyLoraAdapters",
           static_cast<napi_property_attributes>(napi_enumerable)),
       InstanceMethod<&LlamaContext::RemoveLoraAdapters>(
           "removeLoraAdapters",
           static_cast<napi_property_attributes>(napi_enumerable)),
       InstanceMethod<&LlamaContext::GetLoadedLoraAdapters>(
           "getLoadedLoraAdapters",
           static_cast<napi_property_attributes>(napi_enumerable)),
       InstanceMethod<&LlamaContext::InitMultimodal>(
           "initMultimodal",
           static_cast<napi_property_attributes>(napi_enumerable)),
       InstanceMethod<&LlamaContext::IsMultimodalEnabled>(
           "isMultimodalEnabled",
           static_cast<napi_property_attributes>(napi_enumerable)),
       InstanceMethod<&LlamaContext::ReleaseMultimodal>(
           "releaseMultimodal",
           static_cast<napi_property_attributes>(napi_enumerable)),
       InstanceMethod<&LlamaContext::Release>(
           "release", static_cast<napi_property_attributes>(napi_enumerable)),
       StaticMethod<&LlamaContext::ModelInfo>(
           "loadModelInfo",
           static_cast<napi_property_attributes>(napi_enumerable)),
       StaticMethod<&LlamaContext::ToggleNativeLog>(
           "toggleNativeLog",
           static_cast<napi_property_attributes>(napi_enumerable))});
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
  params.model.path = get_option<std::string>(options, "model", "");
  if (params.model.path.empty()) {
    Napi::TypeError::New(env, "Model is required").ThrowAsJavaScriptException();
  }

  params.vocab_only = get_option<bool>(options, "vocab_only", false);
  if (params.vocab_only) {
    params.warmup = false;
  }

  params.chat_template = get_option<std::string>(options, "chat_template", "");

  std::string reasoning_format = get_option<std::string>(options, "reasoning_format", "none");
  if (reasoning_format == "deepseek") {
    params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
  } else {
    params.reasoning_format = COMMON_REASONING_FORMAT_NONE;
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
  params.ctx_shift = get_option<bool>(options, "ctx_shift", true);

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

  auto ctx = sess->context();
  auto model = sess->model();

  std::vector<common_adapter_lora_info> lora;
  auto lora_path = get_option<std::string>(options, "lora", "");
  auto lora_scaled = get_option<float>(options, "lora_scaled", 1.0f);
  if (lora_path != "") {
    common_adapter_lora_info la;
    la.path = lora_path;
    la.scale = lora_scaled;
    la.ptr = llama_adapter_lora_init(model, lora_path.c_str());
    if (la.ptr == nullptr) {
      Napi::TypeError::New(env, "Failed to load lora adapter")
          .ThrowAsJavaScriptException();
    }
    lora.push_back(la);
  }

  if (options.Has("lora_list") && options.Get("lora_list").IsArray()) {
    auto lora_list = options.Get("lora_list").As<Napi::Array>();
    if (lora_list != nullptr) {
      int lora_list_size = lora_list.Length();
      for (int i = 0; i < lora_list_size; i++) {
        auto lora_adapter = lora_list.Get(i).As<Napi::Object>();
        auto path = lora_adapter.Get("path").ToString();
        if (path != nullptr) {
          common_adapter_lora_info la;
          la.path = path;
          la.scale = lora_adapter.Get("scaled").ToNumber().FloatValue();
          la.ptr = llama_adapter_lora_init(model, path.Utf8Value().c_str());
          if (la.ptr == nullptr) {
            Napi::TypeError::New(env, "Failed to load lora adapter")
                .ThrowAsJavaScriptException();
          }
          lora.push_back(la);
        }
      }
    }
  }
  common_set_adapter_lora(ctx, lora);
  _lora = lora;

  _sess = sess;
  _info = common_params_get_system_info(params);

  _templates = common_chat_templates_init(model, params.chat_template);
}

// getSystemInfo(): string
Napi::Value LlamaContext::GetSystemInfo(const Napi::CallbackInfo &info) {
  return Napi::String::New(info.Env(), _info);
}

bool validateModelChatTemplate(const struct llama_model * model, const bool use_jinja, const char * name) {
  const char * tmpl = llama_model_chat_template(model, name);
  if (tmpl == nullptr) {
    return false;
  }
  return common_chat_verify_template(tmpl, use_jinja);
}

static Napi::FunctionReference _log_callback;

// toggleNativeLog(enable: boolean, callback: (log: string) => void): void
void LlamaContext::ToggleNativeLog(const Napi::CallbackInfo &info) {
  bool enable = info[0].ToBoolean().Value();
  if (enable) {
    _log_callback.Reset(info[1].As<Napi::Function>());
    
    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
      llama_log_callback_default(level, text, user_data);

      std::string level_str = "";
      if (level == GGML_LOG_LEVEL_ERROR) {
        level_str = "error";
      } else if (level == GGML_LOG_LEVEL_INFO) {
        level_str = "info";
      } else if (level == GGML_LOG_LEVEL_WARN) {
        level_str = "warn";
      }

      if (_log_callback.IsEmpty()) {
        return;
      }
      try {
        Napi::Env env = _log_callback.Env();
        Napi::HandleScope scope(env);
        _log_callback.Call({
          Napi::String::New(env, level_str),
          Napi::String::New(env, text)
        });
      } catch (const std::exception &e) {
        // printf("Error calling log callback: %s\n", e.what());
      }
    }, nullptr);
  } else {
    _log_callback.Reset();
    llama_log_set(llama_log_callback_default, nullptr);
  }
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
    char val[4096];
    llama_model_meta_val_str_by_index(model, i, val, sizeof(val));

    metadata.Set(key, val);
  }
  Napi::Object details = Napi::Object::New(info.Env());
  details.Set("desc", desc);
  details.Set("nEmbd", llama_model_n_embd(model));
  details.Set("nParams", llama_model_n_params(model));
  details.Set("size", llama_model_size(model));

  Napi::Object chatTemplates = Napi::Object::New(info.Env());
  chatTemplates.Set("llamaChat", validateModelChatTemplate(model, false, ""));
  Napi::Object minja = Napi::Object::New(info.Env());
  minja.Set("default", validateModelChatTemplate(model, true, ""));
  Napi::Object defaultCaps = Napi::Object::New(info.Env());
  defaultCaps.Set("tools", _templates.get()->template_default->original_caps().supports_tools);
  defaultCaps.Set("toolCalls", _templates.get()->template_default->original_caps().supports_tool_calls);
  defaultCaps.Set("toolResponses", _templates.get()->template_default->original_caps().supports_tool_responses);
  defaultCaps.Set("systemRole", _templates.get()->template_default->original_caps().supports_system_role);
  defaultCaps.Set("parallelToolCalls", _templates.get()->template_default->original_caps().supports_parallel_tool_calls);
  defaultCaps.Set("toolCallId", _templates.get()->template_default->original_caps().supports_tool_call_id);
  minja.Set("defaultCaps", defaultCaps);
  minja.Set("toolUse", validateModelChatTemplate(model, true, "tool_use"));
  if (_templates.get()->template_tool_use) {
    Napi::Object toolUseCaps = Napi::Object::New(info.Env());
    toolUseCaps.Set("tools", _templates.get()->template_tool_use->original_caps().supports_tools);
    toolUseCaps.Set("toolCalls", _templates.get()->template_tool_use->original_caps().supports_tool_calls);
    toolUseCaps.Set("toolResponses", _templates.get()->template_tool_use->original_caps().supports_tool_responses);
    toolUseCaps.Set("systemRole", _templates.get()->template_tool_use->original_caps().supports_system_role);
    toolUseCaps.Set("parallelToolCalls", _templates.get()->template_tool_use->original_caps().supports_parallel_tool_calls);
    toolUseCaps.Set("toolCallId", _templates.get()->template_tool_use->original_caps().supports_tool_call_id);
    minja.Set("toolUseCaps", toolUseCaps);
  }
  chatTemplates.Set("minja", minja);
  details.Set("chatTemplates", chatTemplates);

  details.Set("metadata", metadata);

  // Deprecated: use chatTemplates.llamaChat instead
  details.Set("isChatTemplateSupported", validateModelChatTemplate(_sess->model(), false, ""));
  return details;
}

common_chat_params getFormattedChatWithJinja(
  const std::shared_ptr<LlamaSession> &sess,
  const common_chat_templates_ptr &templates,
  const std::string &messages,
  const std::string &chat_template,
  const std::string &json_schema,
  const std::string &tools,
  const bool &parallel_tool_calls,
  const std::string &tool_choice
) {
  common_chat_templates_inputs inputs;
  inputs.messages = common_chat_msgs_parse_oaicompat(json::parse(messages));
  auto useTools = !tools.empty();
  if (useTools) {
    inputs.tools = common_chat_tools_parse_oaicompat(json::parse(tools));
  }
  inputs.parallel_tool_calls = parallel_tool_calls;
  if (!tool_choice.empty()) {
    inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(tool_choice);
  }
  if (!json_schema.empty()) {
    inputs.json_schema = json::parse(json_schema);
  }
  inputs.extract_reasoning = sess->params().reasoning_format != COMMON_REASONING_FORMAT_NONE;

  // If chat_template is provided, create new one and use it (probably slow)
  if (!chat_template.empty()) {
      auto tmps = common_chat_templates_init(sess->model(), chat_template);
      return common_chat_templates_apply(tmps.get(), inputs);
  } else {
      return common_chat_templates_apply(templates.get(), inputs);
  }
}

std::string getFormattedChat(
  const struct llama_model * model,
  const common_chat_templates_ptr &templates,
  const std::string &messages,
  const std::string &chat_template
) {
  common_chat_templates_inputs inputs;
  inputs.messages = common_chat_msgs_parse_oaicompat(json::parse(messages));
  inputs.use_jinja = false;

  // If chat_template is provided, create new one and use it (probably slow)
  if (!chat_template.empty()) {
    auto tmps = common_chat_templates_init(model, chat_template);
    return common_chat_templates_apply(tmps.get(), inputs).prompt;
  } else {
    return common_chat_templates_apply(templates.get(), inputs).prompt;
  }
}

// getFormattedChat(
//   messages: [{ role: string, content: string }],
//   chat_template: string,
//   params: { jinja: boolean, json_schema: string, tools: string, parallel_tool_calls: boolean, tool_choice: string }
// ): object | string
Napi::Value LlamaContext::GetFormattedChat(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  if (info.Length() < 1 || !info[0].IsArray()) {
    Napi::TypeError::New(env, "Array expected").ThrowAsJavaScriptException();
  }
  auto messages = json_stringify(info[0].As<Napi::Array>());
  auto chat_template = info[1].IsString() ? info[1].ToString().Utf8Value() : "";

  auto has_params = info.Length() >= 2;
  auto params = has_params ? info[2].As<Napi::Object>() : Napi::Object::New(env);

  if (get_option<bool>(params, "jinja", false)) {
    std::string json_schema_str = "";
    if (!is_nil(params.Get("response_format"))) {
      auto response_format = params.Get("response_format").As<Napi::Object>();
      auto response_format_type = get_option<std::string>(response_format, "type", "text");
      if (response_format_type == "json_schema" && response_format.Has("json_schema")) {
        auto json_schema = response_format.Get("json_schema").As<Napi::Object>();
        json_schema_str = json_schema.Has("schema") ?
          json_stringify(json_schema.Get("schema").As<Napi::Object>()) :
          "{}";
      } else if (response_format_type == "json_object") {
        json_schema_str = response_format.Has("schema") ?
          json_stringify(response_format.Get("schema").As<Napi::Object>()) :
          "{}";
      }
    }
    auto tools_str = params.Has("tools") ?
      json_stringify(params.Get("tools").As<Napi::Array>()) :
      "";
    auto parallel_tool_calls = get_option<bool>(params, "parallel_tool_calls", false);
    auto tool_choice = get_option<std::string>(params, "tool_choice", "");

    auto chatParams = getFormattedChatWithJinja(_sess, _templates, messages, chat_template, json_schema_str, tools_str, parallel_tool_calls, tool_choice);
    
    Napi::Object result = Napi::Object::New(env);
    result.Set("prompt", chatParams.prompt);
    // chat_format: int
    result.Set("chat_format", static_cast<int>(chatParams.format));
    // grammar: string
    result.Set("grammar", chatParams.grammar);
    // grammar_lazy: boolean
    result.Set("grammea_lazy", chatParams.grammar_lazy);
    // grammar_triggers: [{ value: string, token: number }]
    Napi::Array grammar_triggers = Napi::Array::New(env);
    for (size_t i = 0; i < chatParams.grammar_triggers.size(); i++) {
        const auto & trigger = chatParams.grammar_triggers[i];
        Napi::Object triggerObj = Napi::Object::New(env);
        triggerObj.Set("type", Napi::Number::New(env, trigger.type));
        triggerObj.Set("value", Napi::String::New(env, trigger.value));
        triggerObj.Set("token", Napi::Number::New(env, trigger.token));
        grammar_triggers.Set(i, triggerObj);
    }
    result.Set("grammar_triggers", grammar_triggers);
    // preserved_tokens: string[]
    Napi::Array preserved_tokens = Napi::Array::New(env);
    for (size_t i = 0; i < chatParams.preserved_tokens.size(); i++) {
        preserved_tokens.Set(i, Napi::String::New(env, chatParams.preserved_tokens[i].c_str()));
    }
    result.Set("preserved_tokens", preserved_tokens);
    // additional_stops: string[]
    Napi::Array additional_stops = Napi::Array::New(env);
    for (size_t i = 0; i < chatParams.additional_stops.size(); i++) {
        additional_stops.Set(i, Napi::String::New(env, chatParams.additional_stops[i].c_str()));
    }
    result.Set("additional_stops", additional_stops);

    return result;
  } else {
    auto formatted = getFormattedChat(_sess->model(), _templates, messages, chat_template);
    return Napi::String::New(env, formatted);
  }
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

  std::vector<std::string> stop_words;
  if (options.Has("stop") && options.Get("stop").IsArray()) {
    auto stop_words_array = options.Get("stop").As<Napi::Array>();
    for (size_t i = 0; i < stop_words_array.Length(); i++) {
      stop_words.push_back(stop_words_array.Get(i).ToString().Utf8Value());
    }
  }

  // Process image_paths parameter
  std::vector<std::string> image_paths;
  if (options.Has("image_paths")) {
    if (options.Get("image_paths").IsArray()) {
      auto image_paths_array = options.Get("image_paths").As<Napi::Array>();
      for (size_t i = 0; i < image_paths_array.Length(); i++) {
        image_paths.push_back(image_paths_array.Get(i).ToString().Utf8Value());
      }
    } else if (options.Get("image_paths").IsString()) {
      image_paths.push_back(options.Get("image_paths").ToString().Utf8Value());
    }
  }

  // Check if multimodal is enabled when image_paths are provided
  if (!image_paths.empty() && !(_has_multimodal && _mtmd_ctx != nullptr)) {
    Napi::Error::New(env, "Multimodal support must be enabled via initMultimodal to use image_paths").ThrowAsJavaScriptException();
    return env.Undefined();
  }

  int32_t chat_format = get_option<int32_t>(options, "chat_format", 0);

  common_params params = _sess->params();
  auto grammar_from_params = get_option<std::string>(options, "grammar", "");
  auto has_grammar_set = !grammar_from_params.empty();
  if (has_grammar_set) {
    params.sampling.grammar = grammar_from_params;
  }

  std::string json_schema_str = "";
  if (options.Has("response_format")) {
    auto response_format = options.Get("response_format").As<Napi::Object>();
    auto response_format_type = get_option<std::string>(response_format, "type", "text");
    if (response_format_type == "json_schema" && response_format.Has("json_schema")) {
      auto json_schema = response_format.Get("json_schema").As<Napi::Object>();
      json_schema_str = json_schema.Has("schema") ?
        json_stringify(json_schema.Get("schema").As<Napi::Object>()) :
        "{}";
    } else if (response_format_type == "json_object") {
      json_schema_str = response_format.Has("schema") ?
        json_stringify(response_format.Get("schema").As<Napi::Object>()) :
        "{}";
    }
  }

  // Handle preserved_tokens from options
  if (options.Has("preserved_tokens")) {
    auto preserved_tokens = options.Get("preserved_tokens").As<Napi::Array>();
    for (size_t i = 0; i < preserved_tokens.Length(); i++) {
      auto token = preserved_tokens.Get(i).ToString().Utf8Value();
      auto ids = common_tokenize(_sess->context(), token, /* add_special= */ false, /* parse_special= */ true);
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

      auto type = static_cast<common_grammar_trigger_type>(trigger_obj.Get("type").ToNumber().Int32Value());
      auto word = trigger_obj.Get("value").ToString().Utf8Value();

      if (type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
        auto ids = common_tokenize(_sess->context(), word, /* add_special= */ false, /* parse_special= */ true);
        if (ids.size() == 1) {
          auto token = ids[0];
          if (std::find(params.sampling.preserved_tokens.begin(), params.sampling.preserved_tokens.end(), (llama_token) token) == params.sampling.preserved_tokens.end()) {
            throw std::runtime_error("Grammar trigger word should be marked as preserved token");
          }
          common_grammar_trigger trigger;
          trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
          trigger.value = word;
          trigger.token = token;
          params.sampling.grammar_triggers.push_back(std::move(trigger));
        } else {
          params.sampling.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
        }
      } else {
        common_grammar_trigger trigger;
        trigger.type = type;
        trigger.value = word;
        if (type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
          auto token = (llama_token) trigger_obj.Get("token").ToNumber().Int32Value();
          trigger.token = token;
        }
        params.sampling.grammar_triggers.push_back(std::move(trigger));
      }
    }
  }

  // Handle grammar_lazy from options
  if (options.Has("grammar_lazy")) {
    params.sampling.grammar_lazy = options.Get("grammar_lazy").ToBoolean().Value();
  }

  if (options.Has("messages") && options.Get("messages").IsArray()) {
    auto messages = options.Get("messages").As<Napi::Array>();
    auto chat_template = get_option<std::string>(options, "chat_template", "");
    auto jinja = get_option<bool>(options, "jinja", false);
    if (jinja) {
      auto tools_str = options.Has("tools") ?
        json_stringify(options.Get("tools").As<Napi::Array>()) :
        "";
      auto parallel_tool_calls = get_option<bool>(options, "parallel_tool_calls", false);
      auto tool_choice = get_option<std::string>(options, "tool_choice", "none");

      auto chatParams = getFormattedChatWithJinja(
        _sess,
        _templates,
        json_stringify(messages),
        chat_template,
        json_schema_str,
        tools_str,
        parallel_tool_calls,
        tool_choice
      );  
      
      params.prompt = chatParams.prompt;

      chat_format = chatParams.format;

      for (const auto & token : chatParams.preserved_tokens) {
        auto ids = common_tokenize(_sess->context(), token, /* add_special= */ false, /* parse_special= */ true);
        if (ids.size() == 1) {
          params.sampling.preserved_tokens.insert(ids[0]);
        }
      }

      if (!has_grammar_set) {
        // grammar param always wins jinja template & json_schema
        params.sampling.grammar = chatParams.grammar;
        params.sampling.grammar_lazy = chatParams.grammar_lazy;
        for (const auto & trigger : chatParams.grammar_triggers) {
          params.sampling.grammar_triggers.push_back(trigger);
        }
        has_grammar_set = true;
      }
      
      for (const auto & stop : chatParams.additional_stops) {
        stop_words.push_back(stop);
      }
    } else {
      auto formatted = getFormattedChat(
        _sess->model(),
        _templates,
        json_stringify(messages),
        chat_template
      );
      params.prompt = formatted;
    }
  } else {
    params.prompt = get_option<std::string>(options, "prompt", "");
  }
  if (params.prompt.empty()) {
    Napi::TypeError::New(env, "Prompt is required")
        .ThrowAsJavaScriptException();
  }

  if (!has_grammar_set && !json_schema_str.empty()) {
    params.sampling.grammar = json_schema_to_grammar(json::parse(json_schema_str));
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
  params.sampling.top_n_sigma = get_option<float>(options, "top_n_sigma", -1.0f);
  params.sampling.ignore_eos = get_option<bool>(options, "ignore_eos", false);
  params.n_keep = get_option<int32_t>(options, "n_keep", 0);
  params.sampling.seed = get_option<int32_t>(options, "seed", LLAMA_DEFAULT_SEED);

  Napi::Function callback;
  if (info.Length() >= 2) {
    callback = info[1].As<Napi::Function>();
  }

  auto *worker =
      new LlamaCompletionWorker(info, _sess, callback, params, stop_words, chat_format, image_paths);
  worker->Queue();
  _wip = worker;
  worker->OnComplete([this]() { _wip = nullptr; });
  return worker->Promise();
}

// stopCompletion(): void
void LlamaContext::StopCompletion(const Napi::CallbackInfo &info) {
  if (_wip != nullptr) {
    _wip->SetStop();
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

// applyLoraAdapters(lora_adapters: [{ path: string, scaled: number }]): void
void LlamaContext::ApplyLoraAdapters(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  std::vector<common_adapter_lora_info> lora;
  auto lora_adapters = info[0].As<Napi::Array>();
  for (size_t i = 0; i < lora_adapters.Length(); i++) {
    auto lora_adapter = lora_adapters.Get(i).As<Napi::Object>();
    auto path = lora_adapter.Get("path").ToString().Utf8Value();
    auto scaled = lora_adapter.Get("scaled").ToNumber().FloatValue();
    common_adapter_lora_info la;
    la.path = path;
    la.scale = scaled;
    la.ptr = llama_adapter_lora_init(_sess->model(), path.c_str());
    if (la.ptr == nullptr) {
      Napi::TypeError::New(env, "Failed to load lora adapter")
          .ThrowAsJavaScriptException();
    }
    lora.push_back(la);
  }
  common_set_adapter_lora(_sess->context(), lora);
  _lora = lora;
}

// removeLoraAdapters(): void
void LlamaContext::RemoveLoraAdapters(const Napi::CallbackInfo &info) {
  _lora.clear();
  common_set_adapter_lora(_sess->context(), _lora);
}

// getLoadedLoraAdapters(): Promise<{ count, lora_adapters: [{ path: string,
// scaled: number }] }>
Napi::Value LlamaContext::GetLoadedLoraAdapters(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::Array lora_adapters = Napi::Array::New(env, _lora.size());
  for (size_t i = 0; i < _lora.size(); i++) {
    Napi::Object lora_adapter = Napi::Object::New(env);
    lora_adapter.Set("path", _lora[i].path);
    lora_adapter.Set("scaled", _lora[i].scale);
    lora_adapters.Set(i, lora_adapter);
  }
  return lora_adapters;
}

// release(): Promise<void>
Napi::Value LlamaContext::Release(const Napi::CallbackInfo &info) {
  auto env = info.Env();
  if (_wip != nullptr) {
    _wip->SetStop();
  }
  if (_sess == nullptr) {
    auto promise = Napi::Promise::Deferred(env);
    promise.Resolve(env.Undefined());
    return promise.Promise();
  }
  
  // Clear the mtmd context reference in the session
  if (_mtmd_ctx != nullptr) {
    _sess->set_mtmd_ctx(nullptr);
  }
  
  auto *worker = new DisposeWorker(info, std::move(_sess));
  worker->Queue();
  return worker->Promise();
}

LlamaContext::~LlamaContext() {
  if (_mtmd_ctx != nullptr) {
    mtmd_free(_mtmd_ctx);
    _mtmd_ctx = nullptr;
    _has_multimodal = false;
  }
}

// initMultimodal(options: { path: string, use_gpu?: boolean }): boolean
Napi::Value LlamaContext::InitMultimodal(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1 || !info[0].IsObject()) {
    Napi::TypeError::New(env, "Object expected for mmproj path").ThrowAsJavaScriptException();
  }

  auto options = info[0].As<Napi::Object>();
  auto mmproj_path = options.Get("path").ToString().Utf8Value();
  auto use_gpu = options.Get("use_gpu").ToBoolean().Value();

  if (mmproj_path.empty()) {
    Napi::TypeError::New(env, "mmproj path is required").ThrowAsJavaScriptException();
  }

  console_log(env, "Initializing multimodal with mmproj path: " + mmproj_path);

  auto model = _sess->model();
  auto ctx = _sess->context();
  if (model == nullptr) {
    Napi::Error::New(env, "Model not loaded").ThrowAsJavaScriptException();
    return Napi::Boolean::New(env, false);
  }

  if (_mtmd_ctx != nullptr) {
    mtmd_free(_mtmd_ctx);
    _mtmd_ctx = nullptr;
    _has_multimodal = false;
  }

  // Initialize mtmd context
  mtmd_context_params mtmd_params = mtmd_context_params_default();
  mtmd_params.use_gpu = use_gpu;
  mtmd_params.print_timings = false;
  mtmd_params.n_threads = _sess->params().cpuparams.n_threads;
  mtmd_params.verbosity = (ggml_log_level)GGML_LOG_LEVEL_INFO;

  console_log(env, format_string("Initializing mtmd context with threads=%d, use_gpu=%d", 
                   mtmd_params.n_threads, mtmd_params.use_gpu ? 1 : 0));

  _mtmd_ctx = mtmd_init_from_file(mmproj_path.c_str(), model, mtmd_params);
  if (_mtmd_ctx == nullptr) {
    Napi::Error::New(env, "Failed to initialize multimodal context").ThrowAsJavaScriptException();
    return Napi::Boolean::New(env, false);
  }

  _has_multimodal = true;
  
  // Share the mtmd context with the session
  _sess->set_mtmd_ctx(_mtmd_ctx);

  // Check if the model uses M-RoPE or non-causal attention
  bool uses_mrope = mtmd_decode_use_mrope(_mtmd_ctx);
  bool uses_non_causal = mtmd_decode_use_non_causal(_mtmd_ctx);
  console_log(env, format_string("Model multimodal properties: uses_mrope=%d, uses_non_causal=%d",
             uses_mrope ? 1 : 0, uses_non_causal ? 1 : 0));

  console_log(env, "Multimodal context initialized successfully with mmproj: " + mmproj_path);
  return Napi::Boolean::New(env, true);
}

// isMultimodalEnabled(): boolean
Napi::Value LlamaContext::IsMultimodalEnabled(const Napi::CallbackInfo &info) {
  return Napi::Boolean::New(info.Env(), _has_multimodal && _mtmd_ctx != nullptr);
}

// releaseMultimodal(): void
void LlamaContext::ReleaseMultimodal(const Napi::CallbackInfo &info) {
  if (_mtmd_ctx != nullptr) {
    // Clear the mtmd context reference in the session
    if (_sess != nullptr) {
      _sess->set_mtmd_ctx(nullptr);
    }
    
    // Free the mtmd context
    mtmd_free(_mtmd_ctx);
    _mtmd_ctx = nullptr;
    _has_multimodal = false;
  }
}
