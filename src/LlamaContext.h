#include "common.hpp"
#include "tools/mtmd/mtmd.h"
#include "tools/mtmd/clip.h"

class LlamaCompletionWorker;

class LlamaContext : public Napi::ObjectWrap<LlamaContext> {
public:
  LlamaContext(const Napi::CallbackInfo &info);
  ~LlamaContext();
  static void ToggleNativeLog(const Napi::CallbackInfo &info);
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
  
  // Multimodal methods
  Napi::Value InitMultimodal(const Napi::CallbackInfo &info);
  Napi::Value IsMultimodalEnabled(const Napi::CallbackInfo &info);
  void ReleaseMultimodal(const Napi::CallbackInfo &info);

  std::string _info;
  Napi::Object _meta;
  LlamaSessionPtr _sess = nullptr;
  common_chat_templates_ptr _templates;
  std::vector<common_adapter_lora_info> _lora;
  LlamaCompletionWorker *_wip = nullptr;
  
  // Multimodal support
  mtmd_context *_mtmd_ctx = nullptr;
  bool _has_multimodal = false;
};
