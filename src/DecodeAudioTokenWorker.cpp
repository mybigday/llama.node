#include "DecodeAudioTokenWorker.h"
#include "tts_utils.h"
#include <vector>

DecodeAudioTokenWorker::DecodeAudioTokenWorker(
    const Napi::CallbackInfo &info, llama_model *model, llama_context *ctx,
    int n_threads, const std::vector<llama_token> &tokens)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _model(model), _ctx(ctx),
      _n_threads(n_threads), _tokens(tokens) {}

void DecodeAudioTokenWorker::Execute() {
  const int n_codes = _tokens.size();
  llama_batch batch = llama_batch_init(n_codes, 0, 1);
  for (size_t i = 0; i < _tokens.size(); ++i) {
    common_batch_add(batch, _tokens[i], i, {0}, true);
  }
  if (batch.n_tokens != n_codes) {
    SetError("batch.n_tokens != n_codes");
    return;
  }
  if (llama_encode(_ctx, batch) != 0) {
    SetError("llama_encode() failed");
    return;
  }
  llama_synchronize(_ctx);
  const int n_embd = llama_model_n_embd(_model);
  const float *embd = llama_get_embeddings(_ctx);
  _result = embd_to_audio(embd, n_codes, n_embd, _n_threads);
  apply_fade(_result, 360); // 0.015s * 24000Hz
}

void DecodeAudioTokenWorker::OnOK() {
  auto result =
      Napi::Float32Array::New(Napi::AsyncWorker::Env(), _result.size());
  memcpy(result.Data(), _result.data(), _result.size() * sizeof(float));
  Napi::Promise::Deferred::Resolve(result);
}

void DecodeAudioTokenWorker::OnError(const Napi::Error &err) {
  Napi::Promise::Deferred::Reject(err.Value());
}
