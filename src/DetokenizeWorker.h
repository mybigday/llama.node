#include "common.hpp"
#include <vector>

class DetokenizeWorker : public Napi::AsyncWorker,
                         public Napi::Promise::Deferred {
public:
  DetokenizeWorker(const Napi::CallbackInfo &info, LlamaSessionPtr &sess,
                   std::vector<llama_token> &tokens);

protected:
  void Execute();
  void OnOK();
  void OnError(const Napi::Error &err);

private:
  LlamaSessionPtr _sess;
  std::vector<llama_token> _tokens;
  std::string _text;
};
