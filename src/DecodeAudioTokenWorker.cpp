#include "DecodeAudioTokenWorker.h"
#include "LlamaContext.h"

DecodeAudioTokenWorker::DecodeAudioTokenWorker(const Napi::CallbackInfo &info,
                                               rnllama::llama_rn_context* rn_ctx, std::vector<int32_t> tokens)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _rn_ctx(rn_ctx), _tokens(tokens) {}

void DecodeAudioTokenWorker::Execute() {
  try {
    if (!_rn_ctx->tts_wrapper) {
      SetError("Vocoder not initialized");
      return;
    }
    
    // Convert to llama_token vector - rn-tts handles token adjustment internally
    std::vector<llama_token> llama_tokens;
    for (const auto& token : _tokens) {
      llama_tokens.push_back(token);
    }
    
    // Use the rn-tts API instead of directly accessing the worker
    _result = _rn_ctx->tts_wrapper->decodeAudioTokens(_rn_ctx, llama_tokens);
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