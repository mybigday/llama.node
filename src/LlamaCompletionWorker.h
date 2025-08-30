#pragma once

#include "common.hpp"
#include "rn-llama.h"
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
  LlamaCompletionWorker(const Napi::CallbackInfo &info, rnllama::llama_rn_context* rn_ctx,
                        Napi::Function callback, common_params params,
                        std::vector<std::string> stop_words,
                        int32_t chat_format,
                        bool thinking_forced_open,
                        std::string reasoning_format,
                        const std::vector<std::string> &media_paths = {},
                        const std::vector<llama_token> &guide_tokens = {},
                        bool has_vocoder = false,
                        rnllama::tts_type tts_type_val = rnllama::UNKNOWN,
                        const std::string &prefill_text = "");

  ~LlamaCompletionWorker();

  Napi::Promise GetPromise() { return Napi::Promise::Deferred::Promise(); }

  void OnComplete(std::function<void()> cb) { _onComplete = cb; }

  void SetStop() { _interrupted = true; }

protected:
  void Execute() override;
  void OnOK() override;
  void OnError(const Napi::Error &err) override;

private:

  rnllama::llama_rn_context* _rn_ctx;
  common_params _params;
  std::vector<std::string> _stop_words;
  int32_t _chat_format;
  bool _thinking_forced_open;
  std::string _reasoning_format;
  std::vector<std::string> _media_paths;
  std::vector<llama_token> _guide_tokens;
  std::string _prefill_text;
  std::function<void()> _onComplete;
  bool _has_callback = false;
  bool _interrupted = false;
  Napi::ThreadSafeFunction _tsfn;
  bool _next_token_uses_guide_token = true;
  bool _has_vocoder;
  rnllama::tts_type _tts_type;
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
    std::vector<llama_token> audio_tokens;
  } _result;
};
