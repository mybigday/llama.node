#include "TokenizeWorker.h"
#include "LlamaContext.h"

TokenizeWorker::TokenizeWorker(const Napi::CallbackInfo &info,
                               LlamaSessionPtr &sess, std::string text, std::vector<std::string> media_paths)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _sess(sess), _text(text), _media_paths(media_paths) {}

void TokenizeWorker::Execute() {
  auto mtmd_ctx = _sess->get_mtmd_ctx();
  if (!_media_paths.empty()) {
    try {
      _result = tokenizeWithMedia(mtmd_ctx, _text, _media_paths);
      mtmd_input_chunks_free(_result.chunks);
    } catch (const std::exception &e) {
      SetError(e.what());
    }
  } else {
    const auto tokens = common_tokenize(_sess->context(), _text, false);
    _result.tokens = tokens;
    _result.has_media = false;
  }
}

void TokenizeWorker::OnOK() {
  Napi::HandleScope scope(Napi::AsyncWorker::Env());
  auto result = Napi::Object::New(Napi::AsyncWorker::Env());
  auto tokens =
      Napi::Int32Array::New(Napi::AsyncWorker::Env(), _result.tokens.size());
  memcpy(tokens.Data(), _result.tokens.data(),
         _result.tokens.size() * sizeof(llama_token));
  result.Set("tokens", tokens);
  result.Set("has_media", _result.has_media);
  if (_result.has_media) {
    auto bitmap_hashes = Napi::Array::New(Napi::AsyncWorker::Env(), _result.bitmap_hashes.size());
    for (size_t i = 0; i < _result.bitmap_hashes.size(); i++) {
      bitmap_hashes.Set(i, _result.bitmap_hashes[i]);
    }
    result.Set("bitmap_hashes", bitmap_hashes);
    auto chunk_pos = Napi::Array::New(Napi::AsyncWorker::Env(), _result.chunk_pos.size());
    for (size_t i = 0; i < _result.chunk_pos.size(); i++) {
      chunk_pos.Set(i, _result.chunk_pos[i]);
    }
    result.Set("chunk_pos", chunk_pos);
    auto chunk_pos_media = Napi::Array::New(Napi::AsyncWorker::Env(), _result.chunk_pos_media.size());
    for (size_t i = 0; i < _result.chunk_pos_media.size(); i++) {
      chunk_pos_media.Set(i, _result.chunk_pos_media[i]);
    }
    result.Set("chunk_pos_media", chunk_pos_media);
  }
  Napi::Promise::Deferred::Resolve(result);
}

void TokenizeWorker::OnError(const Napi::Error &err) {
  Napi::Promise::Deferred::Reject(err.Value());
}
