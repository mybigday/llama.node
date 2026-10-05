#include "common.hpp"
#include "rn-llama.h"
#include <string>

class DecideWorker : public Napi::AsyncWorker,
                     public Napi::Promise::Deferred {
public:
  DecideWorker(const Napi::CallbackInfo &info, rnllama::llama_rn_context* rn_ctx,
               std::string request);

protected:
  void Execute();
  void OnOK();
  void OnError(const Napi::Error &err);

private:
  rnllama::llama_rn_context* _rn_ctx;
  std::string _request;
  std::string _result;
};
