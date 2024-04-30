#include "common.hpp"

class DisposeWorker : public Napi::AsyncWorker, public Napi::Promise::Deferred {
public:
  DisposeWorker(const Napi::CallbackInfo &info, LlamaSessionPtr sess);

protected:
  void Execute();
  void OnOK();
  void OnError(const Napi::Error &err);

private:
  LlamaSessionPtr sess_;
};
