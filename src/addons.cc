#include "LlamaContext.h"
#include <napi.h>

// Forward declaration of our cleanup function
extern "C" void cleanup_logging();

// Register cleanup function on module unload
static Napi::Value register_cleanup(const Napi::CallbackInfo &info) {
  napi_add_env_cleanup_hook(
      info.Env(), [](void *) { cleanup_logging(); }, nullptr);

  return info.Env().Undefined();
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  LlamaContext::Init(env, exports);

  // Register our cleanup handler for module unload
  exports.Set("__registerCleanup", Napi::Function::New(env, register_cleanup));

  // Also register cleanup directly on module init
  napi_add_env_cleanup_hook(env, [](void *) { cleanup_logging(); }, nullptr);

  return exports;
}

NODE_API_MODULE(addons, Init)
