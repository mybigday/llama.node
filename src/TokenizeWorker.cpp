#include "TokenizeWorker.h"
#include "LlamaContext.h"

TokenizeWorker::TokenizeWorker(const Napi::CallbackInfo &info,
                               rnllama::llama_rn_context* rn_ctx, std::string text,
                               std::vector<std::string> media_paths)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _rn_ctx(rn_ctx), _text(text),
      _media_paths(media_paths) {}

void TokenizeWorker::Execute() {
  try {
    // Use rn-llama tokenize API directly
    auto result = _rn_ctx->tokenize(_text, _media_paths);
    
    // Convert llama_token to int32_t
    _result.tokens.resize(result.tokens.size());
    for (size_t i = 0; i < result.tokens.size(); i++) {
      _result.tokens[i] = static_cast<int32_t>(result.tokens[i]);
    }
    
    _result.has_media = result.has_media;
    _result.bitmap_hashes = result.bitmap_hashes;
    _result.chunk_pos = result.chunk_pos;
    _result.chunk_pos_media = result.chunk_pos_media;
  } catch (const std::exception &e) {
    SetError(e.what());
  }
}

void TokenizeWorker::OnOK() {
  Napi::Env env = Napi::AsyncWorker::Env();
  Napi::Object ret = Napi::Object::New(env);
  auto tokens = Napi::Int32Array::New(env, _result.tokens.size());
  memcpy(tokens.Data(), _result.tokens.data(), _result.tokens.size() * sizeof(int32_t));
  ret.Set("tokens", tokens);
  ret.Set("has_media", Napi::Boolean::New(env, _result.has_media));
  
  auto bitmap_hashes = Napi::Array::New(env, _result.bitmap_hashes.size());
  for (size_t i = 0; i < _result.bitmap_hashes.size(); i++) {
    bitmap_hashes.Set(i, Napi::String::New(env, _result.bitmap_hashes[i]));
  }
  ret.Set("bitmap_hashes", bitmap_hashes);
  
  auto chunk_pos = Napi::Array::New(env, _result.chunk_pos.size());
  for (size_t i = 0; i < _result.chunk_pos.size(); i++) {
    chunk_pos.Set(i, Napi::Number::New(env, static_cast<double>(_result.chunk_pos[i])));
  }
  ret.Set("chunk_pos", chunk_pos);
  
  auto chunk_pos_media = Napi::Array::New(env, _result.chunk_pos_media.size());
  for (size_t i = 0; i < _result.chunk_pos_media.size(); i++) {
    chunk_pos_media.Set(i, Napi::Number::New(env, static_cast<double>(_result.chunk_pos_media[i])));
  }
  ret.Set("chunk_pos_media", chunk_pos_media);
  
  Napi::Promise::Deferred::Resolve(ret);
}

void TokenizeWorker::OnError(const Napi::Error &err) {
  Napi::Promise::Deferred::Reject(err.Value());
}