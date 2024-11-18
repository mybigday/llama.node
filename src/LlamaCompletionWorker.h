#include "common.hpp"
#include <functional>

struct CompletionResult {
  std::string text = "";
  bool truncated = false;
  size_t tokens_predicted = 0;
  size_t tokens_evaluated = 0;
};

class LlamaCompletionWorker : public Napi::AsyncWorker,
                              public Napi::Promise::Deferred {
public:
  LlamaCompletionWorker(const Napi::CallbackInfo &info, LlamaSessionPtr &sess,
                        Napi::Function callback, common_params params,
                        std::vector<std::string> stop_words = {});

  ~LlamaCompletionWorker();

  inline void Stop() { _stop = true; }

  inline void onComplete(std::function<void()> cb) { _onComplete = cb; }

protected:
  void Execute();
  void OnOK();
  void OnError(const Napi::Error &err);

private:
  LlamaSessionPtr _sess;
  common_params _params;
  std::vector<std::string> _stop_words;
  Napi::ThreadSafeFunction _tsfn;
  bool _has_callback = false;
  bool _stop = false;
  std::function<void()> _onComplete;
  CompletionResult _result;
};
