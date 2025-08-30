#include "common.hpp"
#include "rn-llama/rn-llama.h"

class LoadSessionWorker : public Napi::AsyncWorker,
                          public Napi::Promise::Deferred {
public:
  LoadSessionWorker(const Napi::CallbackInfo &info, rnllama::llama_rn_context* rn_ctx);

protected:
  void Execute();
  void OnOK();
  void OnError(const Napi::Error &err);

private:
  std::string _path;
  rnllama::llama_rn_context* _rn_ctx;
  size_t count = 0;
};
