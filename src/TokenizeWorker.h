#include "common.hpp"
#include "rn-llama/rn-llama.h"
#include <vector>

struct TokenizeResult {
  std::vector<int32_t> tokens;
  bool has_media;
  std::vector<std::string> bitmap_hashes;
  std::vector<size_t> chunk_pos;
  std::vector<size_t> chunk_pos_media;
};

class TokenizeWorker : public Napi::AsyncWorker,
                       public Napi::Promise::Deferred {
public:
  TokenizeWorker(const Napi::CallbackInfo &info, rnllama::llama_rn_context* rn_ctx,
                 std::string text, std::vector<std::string> media_paths);

protected:
  void Execute();
  void OnOK();
  void OnError(const Napi::Error &err);

private:
  rnllama::llama_rn_context* _rn_ctx;
  std::string _text;
  std::vector<std::string> _media_paths;
  TokenizeResult _result;
};