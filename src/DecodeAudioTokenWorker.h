#include "common.hpp"
#include "rn-llama/rn-llama.h"
#include <vector>

class DecodeAudioTokenWorker : public Napi::AsyncWorker,
                               public Napi::Promise::Deferred {
public:
  // Token flow: decodeAudioTokens
  DecodeAudioTokenWorker(const Napi::CallbackInfo &info, rnllama::llama_rn_context* rn_ctx,
                         std::vector<int32_t> tokens);
  // Continuous-latent flow: decodeAudioEmbeddings
  DecodeAudioTokenWorker(const Napi::CallbackInfo &info, rnllama::llama_rn_context* rn_ctx,
                         std::vector<float> embeddings, int embedding_dim);

protected:
  void Execute();
  void OnOK();
  void OnError(const Napi::Error &err);

private:
  rnllama::llama_rn_context* _rn_ctx;
  std::vector<int32_t> _tokens;
  std::vector<float> _embeddings;
  int _embedding_dim = 0;
  bool _is_embeddings = false;
  std::vector<float> _result;
};
