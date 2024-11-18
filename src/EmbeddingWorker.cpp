#include "EmbeddingWorker.h"
#include "LlamaContext.h"

EmbeddingWorker::EmbeddingWorker(const Napi::CallbackInfo &info,
                                 LlamaSessionPtr &sess, std::string text)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _sess(sess), _text(text) {}

void EmbeddingWorker::Execute() {
  llama_kv_cache_clear(_sess->context());
  auto tokens = ::common_tokenize(_sess->context(), _text, true);
  // add SEP if not present
  if (tokens.empty() || tokens.back() != llama_token_sep(_sess->model())) {
    tokens.push_back(llama_token_sep(_sess->model()));
  }
  const int n_embd = llama_n_embd(_sess->model());
  do {
    int ret =
        llama_decode(_sess->context(),
                     llama_batch_get_one(tokens.data(), tokens.size()));
    if (ret < 0) {
      SetError("Failed to inference, code: " + std::to_string(ret));
      break;
    }
    const float *embd = llama_get_embeddings_seq(_sess->context(), 0);
    if (embd == nullptr) {
      SetError("Failed to get embeddings");
      break;
    }
    _result.embedding.resize(n_embd);
    memcpy(_result.embedding.data(), embd, n_embd * sizeof(float));
  } while (false);
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
