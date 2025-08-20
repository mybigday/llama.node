#pragma once

#include "chat.h"
#include "common/common.h"
#include "common/sampling.h"
#include "llama.h"
#include "tools/mtmd/clip.h"
#include "tools/mtmd/mtmd.h"
#include "tools/mtmd/mtmd-helper.h"
#include <memory>
#include <mutex>
#include <napi.h>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

typedef std::unique_ptr<common_sampler, decltype(&common_sampler_free)>
    LlamaCppSampling;
typedef std::unique_ptr<llama_batch, decltype(&llama_batch_free)> LlamaCppBatch;

static bool is_nil(const Napi::Value &value) {
  return value.IsNull() || value.IsUndefined();
}

static std::string json_stringify(const Napi::Object &obj) {
  Napi::Env env = obj.Env();
  Napi::Object json = env.Global().Get("JSON").As<Napi::Object>();
  Napi::Function stringify = json.Get("stringify").As<Napi::Function>();
  return stringify.Call(json, {obj}).As<Napi::String>().ToString();
}

static void console_log(Napi::Env env, const std::string &message) {
  Napi::Function consoleLog = env.Global()
                                  .Get("console")
                                  .As<Napi::Object>()
                                  .Get("log")
                                  .As<Napi::Function>();
  consoleLog.Call({Napi::String::New(env, message)});
}

template <typename T>
constexpr T get_option(const Napi::Object &options, const std::string &name,
                       const T default_value) {
  if (options.Has(name) && !is_nil(options.Get(name))) {
    if constexpr (std::is_same<T, std::string>::value) {
      return options.Get(name).ToString().operator T();
    } else if constexpr (std::is_same<T, int32_t>::value ||
                         std::is_same<T, uint32_t>::value ||
                         std::is_same<T, float>::value ||
                         std::is_same<T, double>::value) {
      return options.Get(name).ToNumber().operator T();
    } else if constexpr (std::is_same<T, bool>::value) {
      return options.Get(name).ToBoolean().operator T();
    } else {
      static_assert(std::is_same<T, std::string>::value ||
                        std::is_same<T, int32_t>::value ||
                        std::is_same<T, uint32_t>::value ||
                        std::is_same<T, float>::value ||
                        std::is_same<T, double>::value ||
                        std::is_same<T, bool>::value,
                    "Unsupported type");
    }
  } else {
    return default_value;
  }
}

class LlamaSession {
public:
  LlamaSession(common_params params) : params_(params) {
    llama_init_ = common_init_from_params(params);
    tokens_.reserve(params.n_ctx);
  }

  ~LlamaSession() { dispose(); }

  inline llama_context *context() { return llama_init_.context.get(); }

  inline llama_model *model() { return llama_init_.model.get(); }

  inline std::vector<llama_token> *tokens_ptr() { return &tokens_; }

  inline void set_tokens(std::vector<llama_token> tokens) {
    tokens_ = std::move(tokens);
  }

  inline std::vector<std::string> *mtmd_bitmap_past_hashes_ptr() {
    return &mtmd_bitmap_past_hashes_;
  }

  inline void set_mtmd_bitmap_past_hashes(std::vector<std::string> hashes) {
    mtmd_bitmap_past_hashes_ = std::move(hashes);
  }

  inline const common_params &params() const { return params_; }

  inline std::mutex &get_mutex() { return mutex; }

  // Getter for the multimodal context
  inline mtmd_context *get_mtmd_ctx() { return _mtmd_ctx; }

  // Setter for the multimodal context
  inline void set_mtmd_ctx(mtmd_context *ctx) { _mtmd_ctx = ctx; }

  void dispose() {
    std::lock_guard<std::mutex> lock(mutex);
    tokens_.clear();

    // mtmd_ctx is owned by LlamaContext, so we don't free it here
    _mtmd_ctx = nullptr;
  }

private:
  common_init_result llama_init_;
  const common_params params_;
  std::vector<llama_token> tokens_{};
  std::vector<std::string> mtmd_bitmap_past_hashes_{};
  std::mutex mutex;
  mtmd_context *_mtmd_ctx = nullptr;
};

typedef std::shared_ptr<LlamaSession> LlamaSessionPtr;

static size_t common_tokens_part(const std::vector<llama_token> &a,
                                 const std::vector<llama_token> &b) {
  size_t i = 0;
  while (i < a.size() && i < b.size() && a[i] == b[i]) {
    i++;
  }
  return i;
}

// Computes FNV-1a hash of the data
static std::string fnv_hash(const uint8_t *data, size_t len) {
  const uint64_t fnv_prime = 0x100000001b3ULL;
  uint64_t hash = 0xcbf29ce484222325ULL;

  for (size_t i = 0; i < len; ++i) {
    hash ^= data[i];
    hash *= fnv_prime;
  }
  return std::to_string(hash);
}

static const std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                        "abcdefghijklmnopqrstuvwxyz"
                                        "0123456789+/";

// Base64 decoding function
static std::vector<uint8_t> base64_decode(const std::string &encoded_string) {
  std::vector<uint8_t> decoded;
  int in_len = encoded_string.size();
  int i = 0;
  int j = 0;
  int in_ = 0;
  unsigned char char_array_4[4], char_array_3[3];

  while (in_len-- && (encoded_string[in_] != '=')) {
    if (isspace(encoded_string[in_])) {
      in_++;
      continue;
    }

    if (encoded_string[in_] == '=' ||
        base64_chars.find(encoded_string[in_]) == std::string::npos) {
      break;
    }

    char_array_4[i++] = encoded_string[in_];
    in_++;
    if (i == 4) {
      for (i = 0; i < 4; i++) {
        char_array_4[i] = base64_chars.find(char_array_4[i]);
      }

      char_array_3[0] =
          (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] =
          ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
      char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

      for (i = 0; i < 3; i++) {
        decoded.push_back(char_array_3[i]);
      }
      i = 0;
    }
  }

  if (i) {
    for (j = i; j < 4; j++) {
      char_array_4[j] = 0;
    }

    for (j = 0; j < 4; j++) {
      char_array_4[j] = base64_chars.find(char_array_4[j]);
    }

    char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
    char_array_3[1] =
        ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
    char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

    for (j = 0; j < i - 1; j++) {
      decoded.push_back(char_array_3[j]);
    }
  }

  return decoded;
}

struct TokenizeResult {
  std::vector<llama_token> tokens;

  bool has_media = false;
  std::vector<std::string> bitmap_hashes;
  std::vector<size_t> chunk_pos;       // both text and media
  std::vector<size_t> chunk_pos_media; // media only
  mtmd_input_chunks *chunks = nullptr;
};

static TokenizeResult
tokenizeWithMedia(mtmd_context *mtmd_ctx, const std::string &prompt,
                  const std::vector<std::string> &media_paths) {
  if (mtmd_ctx == nullptr) {
    throw std::runtime_error("Multimodal context is not initialized");
  }

  TokenizeResult result;
  result.has_media = !media_paths.empty();

  mtmd::bitmaps bitmaps;

  // Load all media paths
  for (const auto &media_path : media_paths) {
    fprintf(
        stdout, "[DEBUG] Loading media: %s\n",
        media_path.substr(0, 50).c_str()); // Only log part of path for base64

    // Check if it's a base64 media
    if (media_path.compare(0, 11, "data:image/") == 0 ||
        media_path.compare(0, 11, "data:audio/") == 0) {

      // Parse base64 data
      std::vector<std::string> parts;
      size_t comma_pos = media_path.find(',');
      if (comma_pos == std::string::npos) {
        result.bitmap_hashes.clear();
        throw std::runtime_error(
            "Invalid base64 media format, missing comma separator");
      }

      std::string header = media_path.substr(0, comma_pos);
      std::string base64_data = media_path.substr(comma_pos + 1);

      if (header.find("base64") == std::string::npos) {
        result.bitmap_hashes.clear();
        throw std::runtime_error("Invalid base64 media");
      }

      // Decode base64
      try {
        // Decode base64 to binary
        std::vector<uint8_t> media_data = base64_decode(base64_data);

        // Load bitmap from memory buffer using direct initialization
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(mtmd_ctx, media_data.data(),
                                                          media_data.size()));
        if (!bmp.ptr) {
          bitmaps.entries.clear();
          throw std::runtime_error("Failed to load base64 media");
        }

        // Calculate bitmap hash (for KV caching)
        std::string hash = fnv_hash(bmp.data(), bmp.n_bytes());
        bmp.set_id(hash.c_str());
        bitmaps.entries.push_back(std::move(bmp));
        result.bitmap_hashes.push_back(hash.c_str());
      } catch (const std::exception &e) {
        bitmaps.entries.clear();
        throw std::runtime_error("Failed to decode base64 media");
      }
    } else if (media_path.compare(0, 7, "http://") == 0 ||
               media_path.compare(0, 8, "https://") == 0) {
      // HTTP URLs are not supported yet
      bitmaps.entries.clear();
      throw std::runtime_error("HTTP/HTTPS URLs are not supported yet");
    } else {
      // Regular file path
      // Check if file exists
      FILE *file = fopen(media_path.c_str(), "rb");
      if (file == nullptr) {
        bitmaps.entries.clear();
        throw std::runtime_error("File does not exist or cannot be opened");
      }

      // Get file size
      fseek(file, 0, SEEK_END);
      long file_size = ftell(file);
      fseek(file, 0, SEEK_SET);
      fclose(file);

      // Create bitmap directly
      mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(mtmd_ctx, media_path.c_str()));
      if (!bmp.ptr) {
        bitmaps.entries.clear();
        throw std::runtime_error("Failed to load media");
      }

      // Calculate bitmap hash (for KV caching)
      std::string hash = fnv_hash(bmp.data(), bmp.nx() * bmp.ny() * 3);
      bmp.set_id(hash.c_str());
      bitmaps.entries.push_back(std::move(bmp));
      result.bitmap_hashes.push_back(hash.c_str());
    }
  }

  result.chunks = mtmd_input_chunks_init();
  if (result.chunks == nullptr) {
    bitmaps.entries.clear();
    throw std::runtime_error("Failed to initialize input chunks");
  }

  // Create input text
  mtmd_input_text input_text;
  input_text.text = prompt.c_str(); // Use the full prompt with media marker
  input_text.add_special = true;   // Add BOS token if this is the first message
  input_text.parse_special = true; // Parse special tokens like <__media__>

  // Tokenize the text and media
  fprintf(stdout, "[DEBUG] Tokenizing text and %zu media\n",
          bitmaps.entries.size());
  auto bitmaps_c_ptr = bitmaps.c_ptr();

  // Cast away const for mtmd_tokenize
  int32_t res =
      mtmd_tokenize(const_cast<mtmd_context *>(mtmd_ctx), result.chunks,
                    &input_text, bitmaps_c_ptr.data(), bitmaps_c_ptr.size());

  if (res != 0) {
    mtmd_input_chunks_free(result.chunks);
    bitmaps.entries.clear();
    throw std::runtime_error("Failed to tokenize text and media");
  }

  // Log chunk information
  size_t num_chunks = mtmd_input_chunks_size(result.chunks);
  fprintf(stdout, "[DEBUG] Tokenization successful: num_chunks=%zu\n",
          num_chunks);

  // Track the total number of tokens (both text and media)
  size_t total_token_count = 0;

  // chunk pos
  for (size_t i = 0; i < num_chunks; i++) {
    result.chunk_pos.push_back(total_token_count);

    const mtmd_input_chunk *chunk = mtmd_input_chunks_get(result.chunks, i);
    mtmd_input_chunk_type chunk_type = mtmd_input_chunk_get_type(chunk);

    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
      size_t n_tokens;
      const llama_token *tokens =
          mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);

      result.tokens.insert(result.tokens.end(), tokens, tokens + n_tokens);
      total_token_count += n_tokens;
    } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ||
               chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
      result.chunk_pos_media.push_back(total_token_count);

      size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
      size_t n_pos = mtmd_input_chunk_get_n_pos(chunk);
      fprintf(stdout, "[DEBUG] Chunk %zu: type=%s, n_tokens=%zu, n_pos=%zu\n",
              i, chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "IMAGE" : "AUDIO",
              n_tokens, n_pos);

      for (size_t j = 0; j < n_pos; j++) {
        result.tokens.push_back(LLAMA_TOKEN_NULL);
      }
      total_token_count += n_pos;
    }
  }

  bitmaps.entries.clear();

  return result;
}

// Process media and add them to the tokenized input
static llama_pos
processMediaPrompt(llama_context *ctx, mtmd_context *mtmd_ctx,
                   LlamaSessionPtr sess, const common_params &params,
                   const std::vector<std::string> &media_paths) {
  if (mtmd_ctx == nullptr) {
    throw std::runtime_error("Multimodal context is not initialized");
  }

  // Multimodal path
  std::string full_prompt = params.prompt;
  auto default_media_marker = mtmd_default_marker();
  // Add media marker if it doesn't already exist
  if (full_prompt.find(default_media_marker) == std::string::npos) {
    full_prompt += " ";
    full_prompt += default_media_marker;
  }

  auto result = tokenizeWithMedia(mtmd_ctx, full_prompt, media_paths);

  auto all_tokens = result.tokens;
  auto chunks = result.chunks;
  auto chunk_pos = result.chunk_pos;
  auto chunk_pos_media = result.chunk_pos_media;
  auto bitmap_hashes = result.bitmap_hashes;

  llama_pos n_past = common_tokens_part(*sess->tokens_ptr(), all_tokens);

  llama_pos new_n_past = n_past;

  // Adjust n_past to position of the text chunk
  // TODO: Edit the text chunk to remove the tokens before n_past to speed up
  // need to update the mtmd api
  auto adjusted_n_past = -1;
  for (size_t i = 0; i < chunk_pos.size(); i++) {
    if (n_past < chunk_pos[i]) {
      break;
    }
    bool is_end = i + 1 == chunk_pos.size();
    if (chunk_pos[i] < n_past && (!is_end && chunk_pos[i + 1] > n_past)
        // is_end & n_past < total_token_count:
        // don't need to adjust and it will skip eval_chunk_single, let
        // nextToken() to finish the job
    ) {
      adjusted_n_past = chunk_pos[i];
    }
  }
  if (adjusted_n_past != -1) {
    n_past = adjusted_n_past;
    new_n_past = n_past;
    fprintf(stdout, "[DEBUG] Adjusted n_past to %d\n", n_past);
  }

  // Compare bitmap hashes, if they are not the same, backtrack n_past to the
  // position of the first mismatch
  auto mtmd_bitmap_past_hashes = sess->mtmd_bitmap_past_hashes_ptr();
  if (mtmd_bitmap_past_hashes->size() > 0) {
    for (size_t i = 0; i < bitmap_hashes.size(); i++) {
      auto pos = chunk_pos_media[i];
      if (n_past < pos) {
        break;
      }
      if (i >= mtmd_bitmap_past_hashes->size()) {
        break;
      }
      if (bitmap_hashes[i] != (*mtmd_bitmap_past_hashes)[i]) {
        n_past = chunk_pos_media[i];
        new_n_past = n_past;
        break;
      }
    }
  }

  // Clear all KV cache entries after position n_past
  auto * kv = llama_get_memory(ctx);
  bool clear_result = llama_memory_seq_rm(kv, 0, n_past, -1);
  if (!clear_result) {
    fprintf(stdout, "[DEBUG] llama_memory_seq_rm failed (likely using a non-Transformer model)! Trying full clear...");
    llama_memory_clear(kv, false);
    n_past = 0;
    new_n_past = n_past;
  }

  size_t num_chunks = mtmd_input_chunks_size(chunks);

  for (size_t i = 0; i < chunk_pos.size(); i++) {
    fprintf(stdout, "[DEBUG] Evaluating chunk %zu: n_past=%d, chunk_pos=%zu\n",
            i, n_past, chunk_pos[i]);

    // Process chunk only if it's after the current n_past
    if (chunk_pos[i] >= new_n_past) {
      bool chunk_logits_last = (i == num_chunks - 1);
      auto chunk = mtmd_input_chunks_get(chunks, i);

      // Cast away const for mtmd_helper_eval_chunk_single
      int32_t res = mtmd_helper_eval_chunk_single(
          const_cast<mtmd_context *>(mtmd_ctx), ctx, chunk, n_past, 0,
          params.n_batch, // batch size
          chunk_logits_last, &new_n_past);

      if (res != 0) {
        mtmd_input_chunks_free(chunks);
        throw std::runtime_error("Failed to process chunk");
      }
      n_past = new_n_past;
    }
  }

  if (n_past == all_tokens.size() && n_past > 0 &&
      all_tokens[n_past - 1] != LLAMA_TOKEN_NULL) {
    // we have to evaluate at least 1 token to generate logits.
    n_past--;
  }

  // Update sampling context to process token sequences
  for (auto &token : all_tokens) {
    if (token == LLAMA_TOKEN_NULL) {
      continue;
    }
  }
  // Set the tokens
  sess->set_tokens(std::move(all_tokens));

  sess->set_mtmd_bitmap_past_hashes(bitmap_hashes);

  // Clean up media resources
  mtmd_input_chunks_free(chunks);
  return n_past;
}