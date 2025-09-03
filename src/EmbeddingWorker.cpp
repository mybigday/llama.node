#include "EmbeddingWorker.h"
#include "LlamaContext.h"

EmbeddingWorker::EmbeddingWorker(const Napi::CallbackInfo &info,
                                 rnllama::llama_rn_context* rn_ctx, std::string text,
                                 common_params &params)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _rn_ctx(rn_ctx), _text(text),
      _params(params) {}

void EmbeddingWorker::Execute() {
  try {
    _rn_ctx->params.prompt = _text;
    _rn_ctx->params.n_predict = 0;

    _result.embedding = _rn_ctx->completion->embedding(_params);
  } catch (const std::exception &e) {
    SetError(e.what());
  }
}

void EmbeddingWorker::OnOK() {
  auto result = Napi::Object::New(Napi::AsyncWorker::Env());
  auto embedding = Napi::Float32Array::New(Napi::AsyncWorker::Env(),
                                           _result.embedding.size());
  memcpy(embedding.Data(), _result.embedding.data(),
         _result.embedding.size() * sizeof(float));
  result.Set("embedding", embedding);
  Napi::Promise::Deferred::Resolve(result);
}

void EmbeddingWorker::OnError(const Napi::Error &err) {
  Napi::Promise::Deferred::Reject(err.Value());
}
