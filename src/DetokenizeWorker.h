#include "common.hpp"
#include "rn-llama/rn-llama.h"
#include <vector>

class DetokenizeWorker : public Napi::AsyncWorker,
                         public Napi::Promise::Deferred {
public:
  DetokenizeWorker(const Napi::CallbackInfo &info, rnllama::llama_rn_context* rn_ctx,
                   std::vector<int32_t> tokens);

protected:
  void Execute();
  void OnOK();
  void OnError(const Napi::Error &err);

private:
  rnllama::llama_rn_context* _rn_ctx;
  std::vector<int32_t> _tokens;
  std::string _text;
};