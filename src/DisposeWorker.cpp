#include "DisposeWorker.h"
#include "rn-completion.h"

DisposeWorker::DisposeWorker(const Napi::CallbackInfo &info,
                             rnllama::llama_rn_context* rn_ctx, rnllama::llama_rn_context** parent_ptr)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _rn_ctx(rn_ctx), _parent_ptr(parent_ptr) {}

void DisposeWorker::Execute() { 
  if (_rn_ctx) {
    // Ensure all child contexts are properly cleaned up first
    try {
      // Now delete the main context
      delete _rn_ctx;
      
      // Set parent pointer to nullptr to prevent double free
      if (_parent_ptr) {
        *_parent_ptr = nullptr;
      }
    } catch (const std::exception& e) {
      SetError(std::string("Error during context disposal: ") + e.what());
      return;
    } catch (...) {
      SetError("Unknown error during context disposal");
      return;
    }
  }
}

void DisposeWorker::OnOK() { Resolve(AsyncWorker::Env().Undefined()); }

void DisposeWorker::OnError(const Napi::Error &err) { Reject(err.Value()); }
