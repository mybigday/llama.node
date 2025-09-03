#include "DetokenizeWorker.h"
#include "LlamaContext.h"

DetokenizeWorker::DetokenizeWorker(const Napi::CallbackInfo &info,
                                   rnllama::llama_rn_context* rn_ctx, std::vector<int32_t> tokens)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _rn_ctx(rn_ctx), _tokens(tokens) {}

void DetokenizeWorker::Execute() {
  const auto text = tokens_to_str(_rn_ctx->ctx, _tokens.begin(), _tokens.end());
  _text = std::move(text);
}

void DetokenizeWorker::OnOK() {
  Napi::Promise::Deferred::Resolve(Napi::String::New(Napi::AsyncWorker::Env(), _text));
}

void DetokenizeWorker::OnError(const Napi::Error &err) {
  Napi::Promise::Deferred::Reject(err.Value());
}