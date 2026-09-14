#include "common.hpp"
#include "rn-llama.h"
#include <vector>

struct RerankResult {
  std::vector<float> scores;
};

class RerankWorker : public Napi::AsyncWorker,
                     public Napi::Promise::Deferred {
public:
  RerankWorker(const Napi::CallbackInfo &info, rnllama::llama_rn_context* rn_ctx,
               std::string query, std::vector<std::string> documents, 
               common_params &params);

protected:
  void Execute();
  void OnOK();
  void OnError(const Napi::Error &err);

private:
  rnllama::llama_rn_context* _rn_ctx;
  std::string _query;
  std::vector<std::string> _documents;
  common_params _params;
  RerankResult _result;
}; 