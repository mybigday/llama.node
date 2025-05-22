#include "TokenizeWorker.h"
#include "LlamaContext.h"

TokenizeWorker::TokenizeWorker(const Napi::CallbackInfo &info,
                               LlamaSessionPtr &sess, std::string text, std::vector<std::string> image_paths)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _sess(sess), _text(text), _image_paths(image_paths) {}

void TokenizeWorker::Execute() {
  auto mtmd_ctx = _sess->get_mtmd_ctx();
  if (!_image_paths.empty()) {
    _result = tokenizeWithImages(mtmd_ctx, _text, _image_paths);
  } else {
    const auto tokens = common_tokenize(_sess->context(), _text, false);
    _result.tokens = tokens;
    _result.has_image = false;
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
  if (_result.has_image) {
    result.Set("has_image", _result.has_image);

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
    auto chunk_pos_images = Napi::Array::New(Napi::AsyncWorker::Env(), _result.chunk_pos_images.size());
    for (size_t i = 0; i < _result.chunk_pos_images.size(); i++) {
      chunk_pos_images.Set(i, _result.chunk_pos_images[i]);
    }
    result.Set("chunk_pos_images", chunk_pos_images);
  }
  Napi::Promise::Deferred::Resolve(result);
}

void TokenizeWorker::OnError(const Napi::Error &err) {
  Napi::Promise::Deferred::Reject(err.Value());
}
