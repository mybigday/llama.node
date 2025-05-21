#include "SaveSessionWorker.h"
#include "LlamaContext.h"

SaveSessionWorker::SaveSessionWorker(const Napi::CallbackInfo &info,
                                     LlamaSessionPtr &sess)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _path(info[0].ToString()),
      _sess(sess) {}

void SaveSessionWorker::Execute() {
  _sess->get_mutex().lock();
  auto tokens = _sess->tokens_ptr();
  auto tokens_to_save = std::vector<llama_token>(tokens->begin(), tokens->end());

  // Find LLAMA_TOKEN_NULL in the tokens and resize the array to the index of the null token
  auto null_token_iter = std::find(tokens_to_save.begin(), tokens_to_save.end(), LLAMA_TOKEN_NULL);
  if (null_token_iter != tokens_to_save.end()) {
    tokens_to_save.resize(std::distance(tokens_to_save.begin(), null_token_iter));
  }

  if (!llama_state_save_file(_sess->context(), _path.c_str(), tokens_to_save.data(),
                             tokens_to_save.size())) {
    SetError("Failed to save session");
  }
  _sess->get_mutex().unlock();
}

void SaveSessionWorker::OnOK() { Resolve(AsyncWorker::Env().Undefined()); }

void SaveSessionWorker::OnError(const Napi::Error &err) { Reject(err.Value()); }
