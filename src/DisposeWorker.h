#include "common.hpp"
#include "rn-llama/rn-llama.h"

class DisposeWorker : public Napi::AsyncWorker, public Napi::Promise::Deferred {
public:
  DisposeWorker(const Napi::CallbackInfo &info, rnllama::llama_rn_context* rn_ctx, rnllama::llama_rn_context** parent_ptr);

protected:
  void Execute();
  void OnOK();
  void OnError(const Napi::Error &err);

private:
  rnllama::llama_rn_context* _rn_ctx;
  rnllama::llama_rn_context** _parent_ptr; // Pointer to the parent's _rn_ctx pointer
};
