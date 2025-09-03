#include "common.hpp"
#include "rn-llama/rn-llama.h"
#include <vector>

struct EmbeddingResult {
  std::vector<float> embedding;
};

class EmbeddingWorker : public Napi::AsyncWorker,
                        public Napi::Promise::Deferred {
public:
  EmbeddingWorker(const Napi::CallbackInfo &info, rnllama::llama_rn_context* rn_ctx,
                  std::string text, common_params &params);

protected:
  void Execute();
  void OnOK();
  void OnError(const Napi::Error &err);

private:
  rnllama::llama_rn_context* _rn_ctx;
  std::string _text;
  common_params _params;
  EmbeddingResult _result;
};
