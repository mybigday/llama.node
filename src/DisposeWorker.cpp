#include "DisposeWorker.h"

DisposeWorker::DisposeWorker(const Napi::CallbackInfo &info,
                             LlamaSessionPtr sess)
    : AsyncWorker(info.Env()), Deferred(info.Env()), sess_(std::move(sess)) {}

void DisposeWorker::Execute() { sess_->dispose(); }

void DisposeWorker::OnOK() { Resolve(AsyncWorker::Env().Undefined()); }

void DisposeWorker::OnError(const Napi::Error &err) { Reject(err.Value()); }
