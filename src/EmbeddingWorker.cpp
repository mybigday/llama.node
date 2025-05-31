#include "EmbeddingWorker.h"
#include "LlamaContext.h"

EmbeddingWorker::EmbeddingWorker(const Napi::CallbackInfo &info,
                                 LlamaSessionPtr &sess, std::string text,
                                 common_params &params)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _sess(sess), _text(text),
      _params(params) {}

void EmbeddingWorker::Execute() {
  llama_kv_self_clear(_sess->context());
  auto tokens = ::common_tokenize(_sess->context(), _text, true);
  // add SEP if not present
  auto vocab = llama_model_get_vocab(_sess->model());
  if (tokens.empty() || tokens.back() != llama_vocab_sep(vocab)) {
    tokens.push_back(llama_vocab_sep(vocab));
  }
  const int n_embd = llama_model_n_embd(_sess->model());
  do {
    auto ctx = _sess->context();
    int ret =
        llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()));
    if (ret < 0) {
      SetError("Failed to inference, code: " + std::to_string(ret));
      break;
    }

    float *embd;
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
      embd = llama_get_embeddings(ctx);
    } else {
      embd = llama_get_embeddings_seq(ctx, 0);
    }
    if (embd == nullptr) {
      SetError("Failed to get embeddings");
      break;
    }
    _result.embedding.resize(n_embd);
    std::vector<float> embedding(embd, embd + n_embd), out(embd, embd + n_embd);
    common_embd_normalize(embedding.data(), out.data(), n_embd,
                          _params.embd_normalize);
    memcpy(_result.embedding.data(), out.data(), n_embd * sizeof(float));
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
