#include "RerankWorker.h"
#include "LlamaContext.h"

RerankWorker::RerankWorker(const Napi::CallbackInfo &info,
                           rnllama::llama_rn_context* rn_ctx, std::string query,
                           std::vector<std::string> documents,
                           common_params &params)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _rn_ctx(rn_ctx), _query(query),
      _documents(documents), _params(params) {}

void RerankWorker::Execute() {
  try {
    std::vector<float> scores = _rn_ctx->completion->rerank(_query, _documents);
    _result.scores = scores;
  } catch (const std::exception &e) {
    SetError(e.what());
  }
}

void RerankWorker::OnOK() {
  Napi::Env env = Napi::AsyncWorker::Env();
  auto result = Napi::Array::New(env, _result.scores.size());
  
  // Create result array with score and index, similar to llama.rn
  for (size_t i = 0; i < _result.scores.size(); i++) {
    auto item = Napi::Object::New(env);
    item.Set("score", Napi::Number::New(env, _result.scores[i]));
    item.Set("index", Napi::Number::New(env, (int)i));
    result.Set(i, item);
  }
  
  Napi::Promise::Deferred::Resolve(result);
}

void RerankWorker::OnError(const Napi::Error &err) {
  Napi::Promise::Deferred::Reject(err.Value());
} 