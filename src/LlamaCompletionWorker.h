#pragma once

#include "common.hpp"
#include <atomic>
#include <functional>
#include <napi.h>

struct CompletionResult {
  std::string text = "";
  bool truncated = false;
  bool context_full = false;
  size_t tokens_predicted = 0;
  size_t tokens_evaluated = 0;
};

class LlamaCompletionWorker : public Napi::AsyncWorker,
                              public Napi::Promise::Deferred {
public:
  LlamaCompletionWorker(const Napi::CallbackInfo &info, LlamaSessionPtr &sess,
                        Napi::Function callback, common_params params,
                        std::vector<std::string> stop_words,
                        int32_t chat_format,
                        std::vector<std::string> image_paths = {});

  ~LlamaCompletionWorker();

  Napi::Promise GetPromise() { return Napi::Promise::Deferred::Promise(); }

  void OnComplete(std::function<void()> cb) {
    _onComplete = cb;
  }

  void SetStop() {
    _stop = true;
  }

protected:
  void Execute() override;
  void OnOK() override;
  void OnError(const Napi::Error &err) override;

private:
  LlamaSessionPtr _sess;
  common_params _params;
  std::vector<std::string> _stop_words;
  int32_t _chat_format;
  std::vector<std::string> _image_paths;
  std::function<void()> _onComplete;
  bool _has_callback = false;
  bool _stop = false;
  Napi::ThreadSafeFunction _tsfn;
  struct {
    size_t tokens_evaluated = 0;
    size_t tokens_predicted = 0;
    bool truncated = false;
    bool context_full = false;
    std::string text;
    bool stopped_eos = false;
    bool stopped_words = false;
    std::string stopping_word;
    bool stopped_limited = false;
  } _result;
};
