#include "EmbeddingWorker.h"
#include "LlamaContext.h"

EmbeddingWorker::EmbeddingWorker(const Napi::CallbackInfo &info,
                                 rnllama::llama_rn_context* rn_ctx, std::string text,
                                 common_params &params)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _rn_ctx(rn_ctx), _text(text),
      _params(params) {}

void EmbeddingWorker::Execute() {
  try {
    // Clear memory and tokenize text
    llama_memory_clear(llama_get_memory(_rn_ctx->ctx), true);
    auto tokens = ::common_tokenize(_rn_ctx->ctx, _text, true);
    
    // Add SEP token if not present
    auto vocab = llama_model_get_vocab(_rn_ctx->model);
    if (tokens.empty() || tokens.back() != llama_vocab_sep(vocab)) {
      tokens.push_back(llama_vocab_sep(vocab));
    }
    
    // Decode tokens to compute embeddings
    int ret = llama_decode(_rn_ctx->ctx, llama_batch_get_one(tokens.data(), tokens.size()));
    if (ret < 0) {
      SetError("Failed to inference, code: " + std::to_string(ret));
      return;
    }
    
    // Get embeddings using rn-completion API
    if (_rn_ctx->completion == nullptr) {
      _rn_ctx->completion = new rnllama::llama_rn_context_completion(_rn_ctx);
    }
    _result.embedding = _rn_ctx->completion->getEmbedding(_params);
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
