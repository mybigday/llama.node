#include "common.hpp"
#include <vector>

struct RerankResult {
  std::vector<float> scores;
};

class RerankWorker : public Napi::AsyncWorker,
                     public Napi::Promise::Deferred {
public:
  RerankWorker(const Napi::CallbackInfo &info, LlamaSessionPtr &sess,
               std::string query, std::vector<std::string> documents, 
               common_params &params);

protected:
  void Execute();
  void OnOK();
  void OnError(const Napi::Error &err);

private:
  LlamaSessionPtr _sess;
  std::string _query;
  std::vector<std::string> _documents;
  common_params _params;
  RerankResult _result;
}; 