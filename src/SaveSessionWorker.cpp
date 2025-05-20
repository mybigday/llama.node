#include "SaveSessionWorker.h"
#include "LlamaContext.h"

SaveSessionWorker::SaveSessionWorker(const Napi::CallbackInfo &info,
                                     LlamaSessionPtr &sess)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _path(info[0].ToString()),
      _sess(sess) {}

void SaveSessionWorker::Execute() {
  _sess->get_mutex().lock();
  auto tokens = _sess->tokens_ptr();

  // Find LLAMA_TOKEN_NULL in the tokens and resize the array to the index of the null token
  auto null_token_iter = std::find(tokens->begin(), tokens->end(), LLAMA_TOKEN_NULL);
  if (null_token_iter != tokens->end()) {
    tokens->resize(std::distance(tokens->begin(), null_token_iter));
  }

  if (!llama_state_save_file(_sess->context(), _path.c_str(), tokens->data(),
                             tokens->size())) {
    SetError("Failed to save session");
  }
  _sess->get_mutex().unlock();
}

void SaveSessionWorker::OnOK() { Resolve(AsyncWorker::Env().Undefined()); }

void SaveSessionWorker::OnError(const Napi::Error &err) { Reject(err.Value()); }
