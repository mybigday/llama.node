#include "common.hpp"
#include <vector>

struct EmbeddingResult {
  std::vector<float> embedding;
};

class EmbeddingWorker : public Napi::AsyncWorker,
                        public Napi::Promise::Deferred {
public:
  EmbeddingWorker(const Napi::CallbackInfo &info, LlamaSessionPtr &sess,
                  std::string text);

protected:
  void Execute();
  void OnOK();
  void OnError(const Napi::Error &err);

private:
  LlamaSessionPtr _sess;
  std::string _text;
  EmbeddingResult _result;
};
