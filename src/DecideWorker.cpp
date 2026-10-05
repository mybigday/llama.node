#include "DecideWorker.h"
#include "LlamaContext.h"

DecideWorker::DecideWorker(const Napi::CallbackInfo &info,
                           rnllama::llama_rn_context* rn_ctx,
                           std::string request)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _rn_ctx(rn_ctx),
      _request(std::move(request)) {}

void DecideWorker::Execute() {
  try {
    // TypeSafe /v1/systemone request in, response out (see rn-decision.h)
    _result = _rn_ctx->decide(json::parse(_request)).dump();
  } catch (const std::exception &e) {
    SetError(e.what());
  }
}

void DecideWorker::OnOK() {
  // JSON.parse keeps the option keys (UTF-8, `__proto__`) as own properties
  Napi::Env env = Napi::AsyncWorker::Env();
  Napi::Promise::Deferred::Resolve(json_parse(env, _result));
}

void DecideWorker::OnError(const Napi::Error &err) {
  Napi::Promise::Deferred::Reject(err.Value());
}
