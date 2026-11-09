#include "DecodeAudioTokenWorker.h"
#include "LlamaContext.h"

DecodeAudioTokenWorker::DecodeAudioTokenWorker(const Napi::CallbackInfo &info,
                                               rnllama::llama_rn_context* rn_ctx, std::vector<int32_t> tokens)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _rn_ctx(rn_ctx), _tokens(tokens) {}

DecodeAudioTokenWorker::DecodeAudioTokenWorker(const Napi::CallbackInfo &info,
                                               rnllama::llama_rn_context* rn_ctx,
                                               std::vector<float> latents)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _rn_ctx(rn_ctx),
      _latents(std::move(latents)), _is_latents(true) {}

void DecodeAudioTokenWorker::Execute() {
  try {
    if (!_rn_ctx->tts_wrapper) {
      SetError("Vocoder not initialized");
      return;
    }

    if (_is_latents) {
      _result = _rn_ctx->tts_wrapper->decodeAudioLatents(_rn_ctx, _latents);
      return;
    }
    _result = _rn_ctx->tts_wrapper->decodeAudioTokens(
        _rn_ctx, std::vector<llama_token>(_tokens.begin(), _tokens.end()));
  } catch (const std::exception &e) {
    SetError(e.what());
  }
}

void DecodeAudioTokenWorker::OnOK() {
  // Create Float32Array and copy the data
  auto result = Napi::Float32Array::New(Napi::AsyncWorker::Env(), _result.size());
  memcpy(result.Data(), _result.data(), _result.size() * sizeof(float));
  Napi::Promise::Deferred::Resolve(result);
}

void DecodeAudioTokenWorker::OnError(const Napi::Error &err) {
  Napi::Promise::Deferred::Reject(err.Value());
}