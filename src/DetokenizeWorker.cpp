#include "DetokenizeWorker.h"
#include "LlamaContext.h"

DetokenizeWorker::DetokenizeWorker(const Napi::CallbackInfo &info,
                                   rnllama::llama_rn_context* rn_ctx, std::vector<int32_t> tokens)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _rn_ctx(rn_ctx), _tokens(tokens) {}

void DetokenizeWorker::Execute() {
  try {
    // Use direct detokenize through rn-llama context
    if (!_rn_ctx->model) {
      SetError("Model not loaded");
      return;
    }
    const llama_vocab *vocab = llama_model_get_vocab(_rn_ctx->model);
    _result.resize(std::max(_result.capacity(), _tokens.size()));
    int32_t n_chars = llama_detokenize(vocab, _tokens.data(), (int32_t)_tokens.size(), &_result[0], (int32_t)_result.size(), false, false);
    if (n_chars < 0) {
      _result.resize(-n_chars);
      n_chars = llama_detokenize(vocab, _tokens.data(), (int32_t)_tokens.size(), &_result[0], (int32_t)_result.size(), false, false);
    }
    if (n_chars >= 0) {
      _result.resize(n_chars);
    }
  } catch (const std::exception &e) {
    SetError(e.what());
  }
}

void DetokenizeWorker::OnOK() {
  Napi::Promise::Deferred::Resolve(Napi::String::New(Napi::AsyncWorker::Env(), _result));
}

void DetokenizeWorker::OnError(const Napi::Error &err) {
  Napi::Promise::Deferred::Reject(err.Value());
}