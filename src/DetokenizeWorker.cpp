#include "DetokenizeWorker.h"
#include "LlamaContext.h"

DetokenizeWorker::DetokenizeWorker(const Napi::CallbackInfo &info,
                                   LlamaSessionPtr &sess,
                                   std::vector<llama_token> &tokens)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _sess(sess),
      _tokens(std::move(tokens)) {}

void DetokenizeWorker::Execute() {
  const auto text = ::llama_detokenize_bpe(_sess->context(), _tokens);
  _text = std::move(text);
}

void DetokenizeWorker::OnOK() {
  Napi::Promise::Deferred::Resolve(
      Napi::String::New(Napi::AsyncWorker::Env(), _text));
}

void DetokenizeWorker::OnError(const Napi::Error &err) {
  Napi::Promise::Deferred::Reject(err.Value());
}
