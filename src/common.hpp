#pragma once

#include "chat.h"
#include "common/common.h"
#include "common/sampling.h"
#include "llama.h"
#include <memory>
#include <napi.h>
#include <string>

typedef std::unique_ptr<common_sampler, decltype(&common_sampler_free)>
    LlamaCppSampling;
typedef std::unique_ptr<llama_batch, decltype(&llama_batch_free)> LlamaCppBatch;

static bool is_nil(const Napi::Value &value) {
  return value.IsNull() || value.IsUndefined();
}

static std::string json_stringify(const Napi::Object &obj) {
  Napi::Env env = obj.Env();
  Napi::Object json = env.Global().Get("JSON").As<Napi::Object>();
  Napi::Function stringify = json.Get("stringify").As<Napi::Function>();
  return stringify.Call(json, {obj}).As<Napi::String>().ToString();
}

static void console_log(Napi::Env env, const std::string &message) {
  Napi::Function consoleLog = env.Global()
                                  .Get("console")
                                  .As<Napi::Object>()
                                  .Get("log")
                                  .As<Napi::Function>();
  consoleLog.Call({Napi::String::New(env, message)});
}

template <typename T>
constexpr T get_option(const Napi::Object &options, const std::string &name,
                       const T default_value) {
  if (options.Has(name) && !is_nil(options.Get(name))) {
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