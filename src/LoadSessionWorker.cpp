#include "LoadSessionWorker.h"
#include "LlamaContext.h"

LoadSessionWorker::LoadSessionWorker(const Napi::CallbackInfo &info,
                                     LlamaSessionPtr &sess)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _path(info[0].ToString()),
      _sess(sess) {}

void LoadSessionWorker::Execute() {
  _sess->get_mutex().lock();
  // reserve the maximum number of tokens for capacity
  std::vector<llama_token> tokens;
  tokens.reserve(_sess->params().n_ctx);

  // Find LLAMA_TOKEN_NULL in the tokens and resize the array to the index of
  // the null token
  auto null_token_iter =
      std::find(tokens.begin(), tokens.end(), LLAMA_TOKEN_NULL);
  if (null_token_iter != tokens.end()) {
    tokens.resize(std::distance(tokens.begin(), null_token_iter));
  }

  if (!llama_state_load_file(_sess->context(), _path.c_str(), tokens.data(),
                             tokens.capacity(), &count)) {
    SetError("Failed to load session");
  }
  tokens.resize(count);
  _sess->set_tokens(std::move(tokens));
  _sess->get_mutex().unlock();
}

void LoadSessionWorker::OnOK() { Resolve(AsyncWorker::Env().Undefined()); }

void LoadSessionWorker::OnError(const Napi::Error &err) { Reject(err.Value()); }
