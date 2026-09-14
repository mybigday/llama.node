#include "common.hpp"
#include "rn-llama.h"

class SaveSessionWorker : public Napi::AsyncWorker,
                          public Napi::Promise::Deferred {
public:
  SaveSessionWorker(const Napi::CallbackInfo &info, rnllama::llama_rn_context* rn_ctx);

protected:
  void Execute();
  void OnOK();
  void OnError(const Napi::Error &err);

private:
  std::string _path;
  rnllama::llama_rn_context* _rn_ctx;
};
