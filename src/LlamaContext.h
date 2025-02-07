#include "common.hpp"

class LlamaCompletionWorker;

class LlamaContext : public Napi::ObjectWrap<LlamaContext> {
public:
  LlamaContext(const Napi::CallbackInfo &info);
  static Napi::Value ModelInfo(const Napi::CallbackInfo& info);
  static void Init(Napi::Env env, Napi::Object &exports);

private:
  Napi::Value GetSystemInfo(const Napi::CallbackInfo &info);
  Napi::Value GetModelInfo(const Napi::CallbackInfo &info);
  Napi::Value GetFormattedChat(const Napi::CallbackInfo &info);
  Napi::Value Completion(const Napi::CallbackInfo &info);
  void StopCompletion(const Napi::CallbackInfo &info);
  Napi::Value Tokenize(const Napi::CallbackInfo &info);
  Napi::Value Detokenize(const Napi::CallbackInfo &info);
  Napi::Value Embedding(const Napi::CallbackInfo &info);
  Napi::Value SaveSession(const Napi::CallbackInfo &info);
  Napi::Value LoadSession(const Napi::CallbackInfo &info);
  void ApplyLoraAdapters(const Napi::CallbackInfo &info);
  void RemoveLoraAdapters(const Napi::CallbackInfo &info);
  Napi::Value GetLoadedLoraAdapters(const Napi::CallbackInfo &info);
  Napi::Value Release(const Napi::CallbackInfo &info);

  std::string _info;
  Napi::Object _meta;
  LlamaSessionPtr _sess = nullptr;
  common_chat_templates _templates;
  std::vector<common_adapter_lora_info> _lora;
  LlamaCompletionWorker *_wip = nullptr;
};
