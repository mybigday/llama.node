#include "common.hpp"
#include <vector>

class DecodeAudioTokenWorker : public Napi::AsyncWorker,
                        public Napi::Promise::Deferred {
public:
  DecodeAudioTokenWorker(const Napi::CallbackInfo &info, llama_model *model, llama_context *ctx, int n_threads,
                  const std::vector<llama_token> &tokens);

protected:
  void Execute();
  void OnOK();
  void OnError(const Napi::Error &err);

private:
  llama_model *_model;
  llama_context *_ctx;
  int _n_threads;
  std::vector<llama_token> _tokens;
  std::vector<float> _result;
};
