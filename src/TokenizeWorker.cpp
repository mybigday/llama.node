#include "TokenizeWorker.h"
#include "LlamaContext.h"

TokenizeWorker::TokenizeWorker(const Napi::CallbackInfo &info,
                               LlamaSessionPtr &sess, std::string text)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _sess(sess), _text(text) {}

void TokenizeWorker::Execute() {
  const auto tokens = ::llama_tokenize(_sess->context(), _text, false);
  _result = {.tokens = std::move(tokens)};
}

void TokenizeWorker::OnOK() {
  Napi::HandleScope scope(Napi::AsyncWorker::Env());
  auto result = Napi::Object::New(Napi::AsyncWorker::Env());
  auto tokens =
      Napi::Int32Array::New(Napi::AsyncWorker::Env(), _result.tokens.size());
  memcpy(tokens.Data(), _result.tokens.data(),
         _result.tokens.size() * sizeof(llama_token));
  result.Set("tokens", tokens);
  Napi::Promise::Deferred::Resolve(result);
}

void TokenizeWorker::OnError(const Napi::Error &err) {
  Napi::Promise::Deferred::Reject(err.Value());
}
