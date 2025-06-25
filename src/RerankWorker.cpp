#include "RerankWorker.h"
#include "LlamaContext.h"

// Helper function to format rerank task: [BOS]query[EOS][SEP]doc[EOS]
static std::vector<llama_token> format_rerank(const llama_vocab * vocab, const std::vector<llama_token> & query, const std::vector<llama_token> & doc) {
    std::vector<llama_token> result;

    // Get EOS token - use SEP token as fallback if EOS is not available
    llama_token eos_token = llama_vocab_eos(vocab);
    if (eos_token == LLAMA_TOKEN_NULL) {
        eos_token = llama_vocab_sep(vocab);
    }

    result.reserve(doc.size() + query.size() + 4);
    if (llama_vocab_get_add_bos(vocab)) {
        result.push_back(llama_vocab_bos(vocab));
    }
    result.insert(result.end(), query.begin(), query.end());
    if (llama_vocab_get_add_eos(vocab)) {
        result.push_back(eos_token);
    }
    if (llama_vocab_get_add_sep(vocab)) {
        result.push_back(llama_vocab_sep(vocab));
    }
    result.insert(result.end(), doc.begin(), doc.end());
    if (llama_vocab_get_add_eos(vocab)) {
        result.push_back(eos_token);
    }

    return result;
}

RerankWorker::RerankWorker(const Napi::CallbackInfo &info,
                           LlamaSessionPtr &sess, std::string query,
                           std::vector<std::string> documents,
                           common_params &params)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _sess(sess), _query(query),
      _documents(documents), _params(params) {}

void RerankWorker::Execute() {
  // Check if this model supports reranking (requires rank pooling type)
  const enum llama_pooling_type pooling_type = llama_pooling_type(_sess->context());
  if (pooling_type != LLAMA_POOLING_TYPE_RANK) {
    SetError("reranking not supported, pooling_type: " + std::to_string(pooling_type));
    return;
  }

  if (!_params.embedding) {
    SetError("embedding disabled but required for reranking");
    return;
  }

  const llama_vocab * vocab = llama_model_get_vocab(_sess->model());
  std::vector<llama_token> query_tokens = ::common_tokenize(_sess->context(), _query, false);

  _result.scores.reserve(_documents.size());

  for (size_t i = 0; i < _documents.size(); ++i) {
    try {
      llama_memory_clear(llama_get_memory(_sess->context()), false);

      const std::string & document = _documents[i];
      std::vector<llama_token> doc_tokens = ::common_tokenize(_sess->context(), document, false);
      std::vector<llama_token> rerank_tokens = format_rerank(vocab, query_tokens, doc_tokens);

      llama_memory_clear(llama_get_memory(_sess->context()), false);

      // Process the rerank input
      int ret = llama_decode(_sess->context(), llama_batch_get_one(rerank_tokens.data(), rerank_tokens.size()));
      if (ret < 0) {
        _result.scores.push_back(-1e6f); // Default low score if computation failed
        continue;
      }

      // Get the rerank score (single embedding value for rank pooling)
      float *data = llama_get_embeddings_seq(_sess->context(), 0);
      if (data) {
        _result.scores.push_back(data[0]); // For rank pooling, the score is the first (and only) dimension
      } else {
        _result.scores.push_back(-1e6f); // Default low score if computation failed
      }

      // Clear KV cache again to prepare for next document
      llama_memory_clear(llama_get_memory(_sess->context()), false);
    } catch (const std::exception &e) {
      _result.scores.push_back(-1e6f);
    }
  }
}

void RerankWorker::OnOK() {
  Napi::Env env = Napi::AsyncWorker::Env();
  auto result = Napi::Array::New(env, _result.scores.size());
  
  // Create result array with score and index, similar to llama.rn
  for (size_t i = 0; i < _result.scores.size(); i++) {
    auto item = Napi::Object::New(env);
    item.Set("score", Napi::Number::New(env, _result.scores[i]));
    item.Set("index", Napi::Number::New(env, (int)i));
    result.Set(i, item);
  }
  
  Napi::Promise::Deferred::Resolve(result);
}

void RerankWorker::OnError(const Napi::Error &err) {
  Napi::Promise::Deferred::Reject(err.Value());
} 