#include "common.hpp"
#include "rn-llama.h"
#include <vector>

class DecodeAudioTokenWorker : public Napi::AsyncWorker,
                               public Napi::Promise::Deferred {
public:
  // Token flow: decodeAudioTokens
  DecodeAudioTokenWorker(const Napi::CallbackInfo &info, rnllama::llama_rn_context* rn_ctx,
                         std::vector<int32_t> tokens);
  // Latent flow: decodeAudioLatents
  DecodeAudioTokenWorker(const Napi::CallbackInfo &info, rnllama::llama_rn_context* rn_ctx,
                         std::vector<float> latents);

protected:
  void Execute();
  void OnOK();
  void OnError(const Napi::Error &err);

private:
  rnllama::llama_rn_context* _rn_ctx;
  std::vector<int32_t> _tokens;
  std::vector<float> _latents;
  bool _is_latents = false;
  std::vector<float> _result;
};
