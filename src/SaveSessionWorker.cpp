#include "SaveSessionWorker.h"
#include "LlamaContext.h"

SaveSessionWorker::SaveSessionWorker(const Napi::CallbackInfo &info,
                                     LlamaSessionPtr &sess)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _path(info[0].ToString()),
      _sess(sess) {}

void SaveSessionWorker::Execute() {
  _sess->get_mutex().lock();
  auto tokens = _sess->tokens_ptr();
  if (!llama_state_save_file(_sess->context(), _path.c_str(), tokens->data(),
                             tokens->size())) {
    SetError("Failed to save session");
  }
  _sess->get_mutex().unlock();
}

void SaveSessionWorker::OnOK() { Resolve(AsyncWorker::Env().Undefined()); }

void SaveSessionWorker::OnError(const Napi::Error &err) { Reject(err.Value()); }
