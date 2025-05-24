#include "common.hpp"
#include <vector>

class TokenizeWorker : public Napi::AsyncWorker,
                       public Napi::Promise::Deferred {
public:
  TokenizeWorker(const Napi::CallbackInfo &info, LlamaSessionPtr &sess,
                 std::string text, std::vector<std::string> media_paths);

protected:
  void Execute();
  void OnOK();
  void OnError(const Napi::Error &err);

private:
  LlamaSessionPtr _sess;
  std::string _text;
  std::vector<std::string> _media_paths;
  TokenizeResult _result;
};
