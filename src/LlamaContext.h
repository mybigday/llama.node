#include "common.hpp"
#include "tools/mtmd/clip.h"
#include "tools/mtmd/mtmd.h"
#include "rn-llama/rn-llama.h"
#include "rn-llama/rn-completion.h"
#include "rn-llama/rn-tts.h"
#include "rn-llama/rn-slot.h"
#include "rn-llama/rn-slot-manager.h"
#include <atomic>
#include <memory>

using namespace rnllama;

class LlamaCompletionWorker;

struct vocoder_context {
  common_params params;
  std::shared_ptr<llama_model> model;
  std::shared_ptr<llama_context> context;
};

class LlamaContext : public Napi::ObjectWrap<LlamaContext> {
public:
  LlamaContext(const Napi::CallbackInfo &info);
  ~LlamaContext();
  static void ToggleNativeLog(const Napi::CallbackInfo &info);
  static Napi::Value ModelInfo(const Napi::CallbackInfo &info);
  static Napi::Value GetBackendDevicesInfo(const Napi::CallbackInfo &info);
  static void Init(Napi::Env env, Napi::Object &exports);

private:
  Napi::Value GetSystemInfo(const Napi::CallbackInfo &info);
  Napi::Value GetModelInfo(const Napi::CallbackInfo &info);
  Napi::Value GetUsedDevices(const Napi::CallbackInfo &info);
  Napi::Value GetFormattedChat(const Napi::CallbackInfo &info);
  Napi::Value Completion(const Napi::CallbackInfo &info);
  void StopCompletion(const Napi::CallbackInfo &info);
  Napi::Value Tokenize(const Napi::CallbackInfo &info);
  Napi::Value Detokenize(const Napi::CallbackInfo &info);
  Napi::Value Embedding(const Napi::CallbackInfo &info);
  Napi::Value Rerank(const Napi::CallbackInfo &info);
  Napi::Value SaveSession(const Napi::CallbackInfo &info);
  Napi::Value LoadSession(const Napi::CallbackInfo &info);
  void ApplyLoraAdapters(const Napi::CallbackInfo &info);
  void RemoveLoraAdapters(const Napi::CallbackInfo &info);
  Napi::Value GetLoadedLoraAdapters(const Napi::CallbackInfo &info);
  Napi::Value Release(const Napi::CallbackInfo &info);

  // Multimodal methods
  Napi::Value InitMultimodal(const Napi::CallbackInfo &info);
  Napi::Value IsMultimodalEnabled(const Napi::CallbackInfo &info);
  Napi::Value GetMultimodalSupport(const Napi::CallbackInfo &info);
  void ReleaseMultimodal(const Napi::CallbackInfo &info);

  // TTS methods
  rnllama::tts_type getTTSType(Napi::Env env, nlohmann::json speaker = nullptr);
  Napi::Value InitVocoder(const Napi::CallbackInfo &info);
  void ReleaseVocoder(const Napi::CallbackInfo &info);
  Napi::Value IsVocoderEnabled(const Napi::CallbackInfo &info);
  Napi::Value GetFormattedAudioCompletion(const Napi::CallbackInfo &info);
  Napi::Value GetAudioCompletionGuideTokens(const Napi::CallbackInfo &info);
  Napi::Value DecodeAudioTokens(const Napi::CallbackInfo &info);

  // Parallel decoding methods
  Napi::Value EnableParallelMode(const Napi::CallbackInfo &info);
  void DisableParallelMode(const Napi::CallbackInfo &info);
  Napi::Value QueueCompletion(const Napi::CallbackInfo &info);
  Napi::Value QueueEmbedding(const Napi::CallbackInfo &info);
  Napi::Value QueueRerank(const Napi::CallbackInfo &info);
  void CancelRequest(const Napi::CallbackInfo &info);

  std::string _info;
  std::vector<std::string> _used_devices;
  Napi::Object _meta;
  LlamaCompletionWorker *_wip = nullptr;

  // Use rn-llama context instead of direct llama.cpp types
  llama_rn_context *_rn_ctx = nullptr;

  // Validity flag for async callbacks to prevent use-after-free
  // Shared pointer ensures callbacks can safely check if context is still alive
  std::shared_ptr<std::atomic<bool>> _context_valid;

  // Progress callback support for model loading
  Napi::ThreadSafeFunction _progress_tsfn;
};
