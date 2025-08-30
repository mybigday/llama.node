#include "SaveSessionWorker.h"
#include "LlamaContext.h"

SaveSessionWorker::SaveSessionWorker(const Napi::CallbackInfo &info,
                                     rnllama::llama_rn_context* rn_ctx)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _path(info[0].ToString()),
      _rn_ctx(rn_ctx) {}

void SaveSessionWorker::Execute() {
  try {
    if (!_rn_ctx || !_rn_ctx->ctx) {
      SetError("Context not available");
      return;
    }

    // For rn-llama, we save the context state directly
    if (_rn_ctx->completion && !_rn_ctx->completion->embd.empty()) {
      auto &tokens = _rn_ctx->completion->embd;
      if (!llama_state_save_file(_rn_ctx->ctx, _path.c_str(),
                                tokens.data(), tokens.size())) {
        SetError("Failed to save session");
      }
    } else {
      // Save empty session if no tokens available
      if (!llama_state_save_file(_rn_ctx->ctx, _path.c_str(), nullptr, 0)) {
        SetError("Failed to save session");
      }
    }
  } catch (const std::exception &e) {
    SetError(e.what());
  }
}

void SaveSessionWorker::OnOK() { Resolve(AsyncWorker::Env().Undefined()); }

void SaveSessionWorker::OnError(const Napi::Error &err) { Reject(err.Value()); }
