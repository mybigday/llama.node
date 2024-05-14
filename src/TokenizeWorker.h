#include "common.hpp"
#include <vector>

struct TokenizeResult {
  std::vector<llama_token> tokens;
};

class TokenizeWorker : public Napi::AsyncWorker,
                       public Napi::Promise::Deferred {
public:
  TokenizeWorker(const Napi::CallbackInfo &info, LlamaSessionPtr &sess,
                 std::string text);

protected:
  void Execute();
  void OnOK();
  void OnError(const Napi::Error &err);

private:
  LlamaSessionPtr _sess;
  std::string _text;
  TokenizeResult _result;
};
