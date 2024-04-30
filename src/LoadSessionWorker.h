#include "common.hpp"

class LoadSessionWorker : public Napi::AsyncWorker,
                          public Napi::Promise::Deferred {
public:
  LoadSessionWorker(const Napi::CallbackInfo &info, LlamaSessionPtr &sess);

protected:
  void Execute();
  void OnOK();
  void OnError(const Napi::Error &err);

private:
  std::string _path;
  LlamaSessionPtr _sess;
  size_t count = 0;
};
