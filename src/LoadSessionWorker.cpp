#include "LoadSessionWorker.h"
#include "LlamaContext.h"

LoadSessionWorker::LoadSessionWorker(const Napi::CallbackInfo &info,
                                     rnllama::llama_rn_context* rn_ctx)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _path(info[0].ToString()),
      _rn_ctx(rn_ctx) {}

void LoadSessionWorker::Execute() {
  try {
    if (!_rn_ctx || !_rn_ctx->ctx) {
      SetError("Context not available");
      return;
    }

    // reserve the maximum number of tokens for capacity
    std::vector<llama_token> tokens;
    tokens.reserve(_rn_ctx->n_ctx);

    if (!llama_state_load_file(_rn_ctx->ctx, _path.c_str(), tokens.data(),
                               tokens.capacity(), &count)) {
      SetError("Failed to load session");
      return;
    }
    
    tokens.resize(count);
    
    // Update the completion context with loaded tokens
    if (_rn_ctx->completion == nullptr) {
      _rn_ctx->completion = new rnllama::llama_rn_context_completion(_rn_ctx);
    }
    _rn_ctx->completion->embd = std::move(tokens);
    _rn_ctx->completion->n_past = count;
  } catch (const std::exception &e) {
    SetError(e.what());
  }
}

void LoadSessionWorker::OnOK() { Resolve(AsyncWorker::Env().Undefined()); }

void LoadSessionWorker::OnError(const Napi::Error &err) { Reject(err.Value()); }
