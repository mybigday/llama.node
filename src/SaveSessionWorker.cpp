#include "SaveSessionWorker.h"
#include "LlamaContext.h"

#include <algorithm>

SaveSessionWorker::SaveSessionWorker(const Napi::CallbackInfo &info,
                                     rnllama::llama_rn_context* rn_ctx)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _path(info[0].ToString()),
      _rn_ctx(rn_ctx) {}

void SaveSessionWorker::Execute() {
  try {
    if (!_rn_ctx || !_rn_ctx->ctx || !_rn_ctx->completion) {
      SetError("Context or completion not initialized");
      return;
    }
    if (_rn_ctx->slot_manager != nullptr) {
      // The single-completion token history does not describe the parallel
      // slots' sequences, so a whole-context save would be inconsistent.
      SetError("Session save is not supported while parallel mode is enabled");
      return;
    }

    // Keep LLAMA_TOKEN_NULL media placeholders: llama_state_save_file
    // serializes the whole memory, so its token list must cover the same
    // positions for the session to remain resumable.
    const auto &tokens = _rn_ctx->completion->embd;

    // Remove stale metadata before overwriting the state file. If the save is
    // interrupted, no metadata is safer than metadata for a different state.
    rnllama::write_state_meta(_path, {});

    if (!llama_state_save_file(_rn_ctx->ctx, _path.c_str(), tokens.data(),
                               tokens.size())) {
      SetError("Failed to save session");
      return;
    }

    // Placeholder tokens identify media positions but not the media itself.
    // Persist the hashes alongside the state so a later load can verify reuse.
    const bool media_retained =
        std::find(tokens.begin(), tokens.end(), LLAMA_TOKEN_NULL) != tokens.end();
    rnllama::write_state_meta(
        _path, media_retained ? _rn_ctx->getMediaHashes()
                              : std::vector<std::string>{});
  } catch (const std::exception &e) {
    SetError(e.what());
  }
}

void SaveSessionWorker::OnOK() { Resolve(AsyncWorker::Env().Undefined()); }

void SaveSessionWorker::OnError(const Napi::Error &err) { Reject(err.Value()); }
