#pragma once

#include "common/common.h"
#include "llama.h"
#include <memory>
#include <mutex>
#include <napi.h>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

typedef std::unique_ptr<llama_model, decltype(&llama_free_model)> LlamaCppModel;
typedef std::unique_ptr<llama_context, decltype(&llama_free)> LlamaCppContext;
typedef std::unique_ptr<llama_sampling_context, decltype(&llama_sampling_free)>
    LlamaCppSampling;
typedef std::unique_ptr<llama_batch, decltype(&llama_batch_free)> LlamaCppBatch;

template <typename T>
constexpr T get_option(const Napi::Object &options, const std::string &name,
                       const T default_value) {
  if (options.Has(name) && !options.Get(name).IsUndefined() &&
      !options.Get(name).IsNull()) {
    if constexpr (std::is_same<T, std::string>::value) {
      return options.Get(name).ToString().operator T();
    } else if constexpr (std::is_same<T, int32_t>::value ||
                         std::is_same<T, uint32_t>::value ||
                         std::is_same<T, float>::value ||
                         std::is_same<T, double>::value) {
      return options.Get(name).ToNumber().operator T();
    } else if constexpr (std::is_same<T, bool>::value) {
      return options.Get(name).ToBoolean().operator T();
    } else {
      static_assert(std::is_same<T, std::string>::value ||
                        std::is_same<T, int32_t>::value ||
                        std::is_same<T, uint32_t>::value ||
                        std::is_same<T, float>::value ||
                        std::is_same<T, double>::value ||
                        std::is_same<T, bool>::value,
                    "Unsupported type");
    }
  } else {
    return default_value;
  }
}

class LlamaSession {
public:
  LlamaSession(llama_model *model, llama_context *ctx, gpt_params params)
      : model_(LlamaCppModel(model, llama_free_model)),
        ctx_(LlamaCppContext(ctx, llama_free)), params_(params) {
    tokens_.reserve(params.n_ctx);
  }

  ~LlamaSession() { dispose(); }

  inline llama_context *context() { return ctx_.get(); }

  inline llama_model *model() { return model_.get(); }

  inline std::vector<llama_token> *tokens_ptr() { return &tokens_; }

  inline void set_tokens(std::vector<llama_token> tokens) {
    tokens_ = std::move(tokens);
  }

  inline const gpt_params &params() const { return params_; }

  inline std::mutex &get_mutex() { return mutex; }

  void dispose() {
    std::lock_guard<std::mutex> lock(mutex);
    tokens_.clear();
    ctx_.reset();
    model_.reset();
  }

private:
  LlamaCppModel model_;
  LlamaCppContext ctx_;
  const gpt_params params_;
  std::vector<llama_token> tokens_{};
  std::mutex mutex;
};

typedef std::shared_ptr<LlamaSession> LlamaSessionPtr;
