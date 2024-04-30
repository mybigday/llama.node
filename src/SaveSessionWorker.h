#include "common.hpp"

class SaveSessionWorker : public Napi::AsyncWorker,
                          public Napi::Promise::Deferred {
public:
  SaveSessionWorker(const Napi::CallbackInfo &info, LlamaSessionPtr &sess);

protected:
  void Execute();
  void OnOK();
  void OnError(const Napi::Error &err);

private:
  std::string _path;
  LlamaSessionPtr _sess;
};
