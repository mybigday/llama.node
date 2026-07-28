#include "LoadSessionWorker.h"
#include "LlamaContext.h"

#include <algorithm>

LoadSessionWorker::LoadSessionWorker(const Napi::CallbackInfo &info,
                                     rnllama::llama_rn_context* rn_ctx)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _path(info[0].ToString()),
      _rn_ctx(rn_ctx) {}

void LoadSessionWorker::Execute() {
  try {
    if (!_rn_ctx || !_rn_ctx->ctx || !_rn_ctx->completion) {
      SetError("Context or completion not initialized");
      return;
    }
    if (_rn_ctx->slot_manager != nullptr) {
      // llama_state_load_file restores every sequence at once, which would
      // corrupt in-flight parallel slots.
      SetError("Session load is not supported while parallel mode is enabled");
      return;
    }

    std::vector<llama_token> tokens(llama_n_ctx(_rn_ctx->ctx));
    if (!llama_state_load_file(_rn_ctx->ctx, _path.c_str(), tokens.data(),
                               tokens.size(), &count)) {
      SetError("Failed to load session");
      return;
    }
    // Keep LLAMA_TOKEN_NULL placeholders: they represent media positions in
    // the restored memory.
    tokens.resize(count);

    // Legacy multimodal or token-limited files may contain a token list that
    // does not match the restored memory. Reconcile it now so the next decode
    // either resumes safely or starts cold instead of failing later.
    auto *memory = llama_get_memory(_rn_ctx->ctx);
    const llama_pos n_tokens = static_cast<llama_pos>(tokens.size());
    const llama_pos pos_max = llama_memory_seq_pos_max(memory, 0);
    const bool tokens_have_media =
        std::find(tokens.begin(), tokens.end(), LLAMA_TOKEN_NULL) != tokens.end();
    const bool mrope_media =
        rnllama::model_uses_mrope(_rn_ctx->model) && tokens_have_media;
    bool resumable =
        pos_max + 1 == n_tokens ||
        (mrope_media && pos_max >= 0 && pos_max + 1 < n_tokens);

    if (!resumable && pos_max + 1 > n_tokens) {
      resumable = llama_memory_seq_rm(memory, 0, n_tokens, -1) &&
                  llama_memory_seq_pos_max(memory, 0) + 1 == n_tokens;
      if (resumable) {
        // A rolled-back SWA cache is only reusable if it still contains the
        // full attention window ending at n_tokens.
        const bool recurrent_or_hybrid =
            llama_model_is_recurrent(_rn_ctx->model) ||
            llama_model_is_hybrid(_rn_ctx->model);
        const int32_t n_swa =
            _rn_ctx->params.swa_full ? 0 : llama_model_n_swa(_rn_ctx->model);
        if (n_swa > 0 && !recurrent_or_hybrid) {
          const llama_pos pos_min = llama_memory_seq_pos_min(memory, 0);
          const llama_pos pos_min_threshold =
              std::max<llama_pos>(0, n_tokens - n_swa);
          resumable = pos_min == 0 ||
                      (pos_min > 0 && pos_min < pos_min_threshold);
        }
      }
    }

    if (!resumable) {
      llama_memory_seq_rm(memory, 0, 0, -1);
      tokens.clear();
    }

    // Missing or malformed metadata fails closed: processMedia will reprocess
    // media rather than reusing memory whose identity cannot be verified.
    _rn_ctx->setMediaHashes(tokens.empty()
                                ? std::vector<std::string>{}
                                : rnllama::read_state_meta(_path));

    count = tokens.size();
    _rn_ctx->completion->embd = std::move(tokens);
    _rn_ctx->completion->n_past = static_cast<llama_pos>(count);
  } catch (const std::exception &e) {
    SetError(e.what());
  }
}

void LoadSessionWorker::OnOK() { Resolve(AsyncWorker::Env().Undefined()); }

void LoadSessionWorker::OnError(const Napi::Error &err) { Reject(err.Value()); }
