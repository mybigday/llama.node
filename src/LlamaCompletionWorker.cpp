#include "LlamaCompletionWorker.h"
#include "LlamaContext.h"

// Computes FNV-1a hash of the data
static std::string fnv_hash(const uint8_t * data, size_t len) {
  const uint64_t fnv_prime = 0x100000001b3ULL;
  uint64_t hash = 0xcbf29ce484222325ULL;

  for (size_t i = 0; i < len; ++i) {
    hash ^= data[i];
    hash *= fnv_prime;
  }
  return std::to_string(hash);
}

static const std::string base64_chars =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
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

    if (encoded_string[in_] == '=' || base64_chars.find(encoded_string[in_]) == std::string::npos) {
      break;
    }

    char_array_4[i++] = encoded_string[in_]; in_++;
    if (i == 4) {
      for (i = 0; i < 4; i++) {
        char_array_4[i] = base64_chars.find(char_array_4[i]);
      }

      char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
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
    char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
    char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

    for (j = 0; j < i - 1; j++) {
      decoded.push_back(char_array_3[j]);
    }
  }

  return decoded;
}

size_t common_part(const std::vector<llama_token> &a,
                   const std::vector<llama_token> &b) {
  size_t i = 0;
  while (i < a.size() && i < b.size() && a[i] == b[i]) {
    i++;
  }
  return i;
}

// Process images and add them to the tokenized input
llama_pos processImage(
  llama_context* ctx,
  const mtmd_context* mtmd_ctx,
  LlamaSessionPtr sess,
  const common_params& params,  
  const std::vector<std::string>& image_paths
) {
  if (mtmd_ctx == nullptr) {
    return false;
  }

  // Multimodal path
  std::string full_prompt = params.prompt;
  // Add image marker if it doesn't already exist
  if (full_prompt.find("<__image__>") == std::string::npos) {
    full_prompt += " <__image__>";
  }

  // Prepare bitmaps array for all images
  mtmd::bitmaps bitmaps;

  std::vector<std::string> bitmap_hashes;

  // Load all images
  for (const auto& image_path : image_paths) {
    fprintf(stdout, "[DEBUG] Loading image: %s\n",
             image_path.substr(0, 50).c_str()); // Only log part of path for base64

    // Check if it's a base64 image
    if (image_path.compare(0, 11, "data:image/") == 0) {

      // Parse base64 data
      std::vector<std::string> parts;
      size_t comma_pos = image_path.find(',');
      if (comma_pos == std::string::npos) {
        bitmaps.entries.clear();
        return false;
      }

      std::string header = image_path.substr(0, comma_pos);
      std::string base64_data = image_path.substr(comma_pos + 1);

      if (header.find("base64") == std::string::npos) {
        bitmaps.entries.clear();
        return false;
      }

      // Decode base64
      try {
        // Decode base64 to binary
        std::vector<uint8_t> image_data = base64_decode(base64_data);

        // Load bitmap from memory buffer using direct initialization
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(image_data.data(), image_data.size()));
        if (!bmp.ptr) {
          bitmaps.entries.clear();
          return false;
        }

        // Calculate bitmap hash (for KV caching)
        std::string hash = fnv_hash(bmp.data(), bmp.nx()*bmp.ny()*3);
        bmp.set_id(hash.c_str());
        bitmaps.entries.push_back(std::move(bmp));
        bitmap_hashes.push_back(hash.c_str());
      } catch (const std::exception& e) {
        bitmaps.entries.clear();
        return false;
      }
    } else if (image_path.compare(0, 7, "http://") == 0 || image_path.compare(0, 8, "https://") == 0) {
      // HTTP URLs are not supported yet
      bitmaps.entries.clear();
      return false;
    } else {
      // Check if file exists
      FILE* file = fopen(image_path.c_str(), "rb");
      if (file == nullptr) {
        bitmaps.entries.clear();
        return false;
      }

      // Get file size
      fseek(file, 0, SEEK_END);
      long file_size = ftell(file);
      fseek(file, 0, SEEK_SET);
      fclose(file);

      // Create bitmap directly
      mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(image_path.c_str()));
      if (!bmp.ptr) {
        bitmaps.entries.clear();
        return false;
      }

      // Calculate bitmap hash (for KV caching)
      std::string hash = fnv_hash(bmp.data(), bmp.nx()*bmp.ny()*3);
      bmp.set_id(hash.c_str());
      bitmaps.entries.push_back(std::move(bmp));
      bitmap_hashes.push_back(hash.c_str());
    }
  }

  mtmd_input_chunks* chunks = mtmd_input_chunks_init();
  if (chunks == nullptr) {
    bitmaps.entries.clear();
    return false;
  }

  // Create input text
  mtmd_input_text input_text;
  input_text.text = full_prompt.c_str(); // Use the full prompt with image marker
  input_text.add_special = true;  // Add BOS token if this is the first message
  input_text.parse_special = true;       // Parse special tokens like <__image__>

  // Tokenize the text and images
  fprintf(stdout, "[DEBUG] Tokenizing text and %zu images\n", bitmaps.entries.size());
  auto bitmaps_c_ptr = bitmaps.c_ptr();
  
  // Cast away const for mtmd_tokenize
  int32_t res = mtmd_tokenize(
    const_cast<mtmd_context*>(mtmd_ctx), 
    chunks, 
    &input_text, 
    bitmaps_c_ptr.data(), 
    bitmaps_c_ptr.size()
  );
  
  if (res != 0) {
    mtmd_input_chunks_free(chunks);
    bitmaps.entries.clear();
    return false;
  }

  // Log chunk information
  size_t num_chunks = mtmd_input_chunks_size(chunks);
  fprintf(stdout, "[DEBUG] Tokenization successful: num_chunks=%zu\n", num_chunks);

  // Create a vector to store all tokens (both text and image)
  std::vector<llama_token> all_tokens;

  // Track the total number of tokens (both text and image)
  size_t total_token_count = 0;

  // chunk pos
  std::vector<size_t> chunk_pos;
  std::vector<size_t> chunk_pos_images;
  for (size_t i = 0; i < num_chunks; i++) {
    chunk_pos.push_back(total_token_count);

    const mtmd_input_chunk* chunk = mtmd_input_chunks_get(chunks, i);
    mtmd_input_chunk_type chunk_type = mtmd_input_chunk_get_type(chunk);

    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
      size_t n_tokens;
      const llama_token* tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);

      all_tokens.insert(all_tokens.end(), tokens, tokens + n_tokens);
      total_token_count += n_tokens;
    } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
      chunk_pos_images.push_back(total_token_count);

      const mtmd_image_tokens* img_tokens = mtmd_input_chunk_get_tokens_image(chunk);
      size_t n_tokens = mtmd_image_tokens_get_n_tokens(img_tokens);
      size_t n_pos = mtmd_image_tokens_get_n_pos(img_tokens);

      for (size_t j = 0; j < n_pos; j++) {
        all_tokens.push_back(LLAMA_TOKEN_NULL);
      }
      total_token_count += n_pos;
    }
  }

  llama_pos n_past = common_part(*sess->tokens_ptr(), all_tokens);

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
    if (
      chunk_pos[i] < n_past &&
      (!is_end && chunk_pos[i + 1] > n_past)
      // is_end & n_past < total_token_count:
      // don't need to adjust and it will skip eval_chunk_single, let nextToken() to finish the job
    ) {
      adjusted_n_past = chunk_pos[i];
    }
  }
  if (adjusted_n_past != -1) {
    n_past = adjusted_n_past;
    new_n_past = n_past;
    fprintf(stdout, "[DEBUG] Adjusted n_past to %d\n", n_past);
  }

  // Compare bitmap hashes, if they are not the same, backtrack n_past to the position of the first mismatch
  auto mtmd_bitmap_past_hashes = sess->mtmd_bitmap_past_hashes_ptr();
  if (mtmd_bitmap_past_hashes->size() > 0) {
    for (size_t i = 0; i < bitmap_hashes.size(); i++) {
      auto pos = chunk_pos_images[i];
      if (n_past < pos) {
        break;
      }
      if (i >= mtmd_bitmap_past_hashes->size()) {
        break;
      }
      if (bitmap_hashes[i] != (*mtmd_bitmap_past_hashes)[i]) {
        n_past = chunk_pos_images[i];
        new_n_past = n_past;
        break;
      }
    }
  }

  // Clear all KV cache entries after position n_past
  llama_kv_self_seq_rm(ctx, 0, n_past, -1);

  for (size_t i = 0; i < chunk_pos.size(); i++) {
    fprintf(stdout, "[DEBUG] Evaluating chunk %zu: n_past=%d, chunk_pos=%zu\n", i, n_past, chunk_pos[i]);

    // Process chunk only if it's after the current n_past
    if (chunk_pos[i] >= new_n_past) {
      bool chunk_logits_last = (i == num_chunks - 1);
      auto chunk = mtmd_input_chunks_get(chunks, i);

      // Cast away const for mtmd_helper_eval_chunk_single
      int32_t res = mtmd_helper_eval_chunk_single(
        const_cast<mtmd_context*>(mtmd_ctx),
        ctx,
        chunk,
        n_past,
        0,
        params.n_batch, // batch size
        chunk_logits_last,
        &new_n_past
      );
      
      if (res != 0) {
        mtmd_input_chunks_free(chunks);
        bitmaps.entries.clear();
        return false;
      }
      n_past = new_n_past;
    }
  }

  if (n_past == total_token_count && n_past > 0 && all_tokens[n_past - 1] != LLAMA_TOKEN_NULL) {
    // we have to evaluate at least 1 token to generate logits.
    n_past--;
  }

  // Update sampling context to process token sequences
  for (auto & token : all_tokens) {
    if (token == LLAMA_TOKEN_NULL) {
      continue;
    }
  }
  // Set the tokens
  sess->set_tokens(std::move(all_tokens));

  sess->set_mtmd_bitmap_past_hashes(bitmap_hashes);

  // Clean up image resources
  mtmd_input_chunks_free(chunks);
  bitmaps.entries.clear();
  return n_past;
}

size_t findStoppingStrings(const std::string &text,
                           const size_t last_token_size,
                           const std::vector<std::string> &stop_words) {
  size_t stop_pos = std::string::npos;

  for (const std::string &word : stop_words) {
    size_t pos;

    const size_t tmp = word.size() + last_token_size;
    const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

    pos = text.find(word, from_pos);

    if (pos != std::string::npos &&
        (stop_pos == std::string::npos || pos < stop_pos)) {
      stop_pos = pos;
    }
  }

  return stop_pos;
}

LlamaCompletionWorker::LlamaCompletionWorker(
    const Napi::CallbackInfo &info, LlamaSessionPtr &sess,
    Napi::Function callback, common_params params,
    std::vector<std::string> stop_words,
    int32_t chat_format,
    std::vector<std::string> image_paths)
    : AsyncWorker(info.Env()), Deferred(info.Env()), _sess(sess),
      _params(params), _stop_words(stop_words), _chat_format(chat_format),
      _image_paths(image_paths) {
  if (!callback.IsEmpty()) {
    _tsfn = Napi::ThreadSafeFunction::New(info.Env(), callback,
                                          "LlamaCompletionCallback", 0, 1);
    _has_callback = true;
  }
}

LlamaCompletionWorker::~LlamaCompletionWorker() {
  if (_has_callback) {
    _tsfn.Release();
  }
}

void LlamaCompletionWorker::Execute() {
  _sess->get_mutex().lock();
  const auto t_main_start = ggml_time_us();
  const size_t n_ctx = _params.n_ctx;
  const auto n_keep = _params.n_keep;
  size_t n_cur = 0;
  size_t n_input = 0;
  const auto model = _sess->model();
  auto vocab = llama_model_get_vocab(model);

  const bool add_bos = llama_vocab_get_add_bos(vocab);
  auto ctx = _sess->context();

  auto sparams = llama_sampler_chain_default_params();

  LlamaCppSampling sampling{common_sampler_init(model, _params.sampling),
                            common_sampler_free};

  // Process images if any are provided
  if (!_image_paths.empty()) {
    const auto* mtmd_ctx = _sess->get_mtmd_ctx();
    
    if (mtmd_ctx != nullptr) {
      // Process the images and get the tokens
      n_cur = processImage(
        ctx,
        mtmd_ctx,
        _sess,
        _params,
        _image_paths
      );
      
      if (n_cur <= 0) {
        SetError("Failed to process images");
        _sess->get_mutex().unlock();
        return;
      }

      fprintf(stdout, "[DEBUG] Image processing successful, n_cur=%zu, tokens=%zu\n", 
                       n_cur, _sess->tokens_ptr()->size());

      n_input = _sess->tokens_ptr()->size();
      if (n_cur == n_input) {
        --n_cur;
      }
      n_input -= n_cur;
    } else {
      SetError("Multimodal context not initialized");
      _sess->get_mutex().unlock();
      return;
    }
  } else {
    // Text-only path
    std::vector<llama_token> prompt_tokens = ::common_tokenize(ctx, _params.prompt, add_bos);
    n_input = prompt_tokens.size();
    
    if (_sess->tokens_ptr()->size() > 0) {
      n_cur = common_part(*(_sess->tokens_ptr()), prompt_tokens);
      if (n_cur == n_input) {
        --n_cur;
      }
      n_input -= n_cur;
      llama_kv_self_seq_rm(ctx, 0, n_cur, -1);
    }
    // Set the tokens
    _sess->set_tokens(std::move(prompt_tokens));
  }

  const int max_len = _params.n_predict < 0 ? 0 : _params.n_predict;
  _sess->tokens_ptr()->reserve(_sess->tokens_ptr()->size() + max_len);

  auto embd = _sess->tokens_ptr();
  for (int i = 0; i < max_len || _stop; i++) {
    // check if we need to remove some tokens
    if (embd->size() >= _params.n_ctx) {
      if (!_params.ctx_shift) {
        // Context is full and ctx_shift is disabled, so we need to stop
        _result.context_full = true;
        break;
      }
      
      const int n_left = n_cur - n_keep - 1;
      const int n_discard = n_left / 2;

      llama_kv_self_seq_rm(ctx, 0, n_keep + 1, n_keep + n_discard + 1);
      llama_kv_self_seq_add(ctx, 0, n_keep + 1 + n_discard, n_cur, -n_discard);

      // shift the tokens
      embd->insert(embd->begin() + n_keep + 1,
                   embd->begin() + n_keep + 1 + n_discard, embd->end());
      embd->resize(embd->size() - n_discard);

      n_cur -= n_discard;
      _result.truncated = true;
    }
    
    // For multimodal input, n_past might already be set
    // Only decode text tokens if we have any input left
    if (n_input > 0) {
      int ret = llama_decode(
          ctx, llama_batch_get_one(embd->data() + n_cur, n_input));
      if (ret < 0) {
        SetError("Failed to decode token, code: " + std::to_string(ret));
        break;
      }
    }
    
    // sample the next token
    const llama_token new_token_id =
        common_sampler_sample(sampling.get(), ctx, -1);
    common_sampler_accept(sampling.get(), new_token_id, true);
    // prepare the next batch
    embd->emplace_back(new_token_id);
    auto token = common_token_to_piece(ctx, new_token_id);
    _result.text += token;
    n_cur += n_input;
    _result.tokens_evaluated += n_input;
    _result.tokens_predicted += 1;
    n_input = 1;
    if (_has_callback) {
      // TODO: When we got possible stop words (startsWith)
      // we should avoid calling the callback, wait for the next token
      const char *c_token = strdup(token.c_str());
      _tsfn.BlockingCall(c_token, [](Napi::Env env, Napi::Function jsCallback,
                                     const char *value) {
        auto obj = Napi::Object::New(env);
        obj.Set("token", Napi::String::New(env, value));
        delete value;
        jsCallback.Call({obj});
      });
    }
    // is it an end of generation?
    if (llama_vocab_is_eog(vocab, new_token_id)) {
      _result.stopped_eos = true;
      // TODO: EOS token should be cut
      break;
    }
    // check for stop words
    if (!_stop_words.empty()) {
      const size_t stop_pos =
          findStoppingStrings(_result.text, token.size(), _stop_words);
      if (stop_pos != std::string::npos) {
        _result.stopped_words = true;
        _result.stopping_word = _result.text.substr(stop_pos, token.size());
        _result.text = _result.text.substr(0, stop_pos - 1);
        break;
      }
    }
  }
  if (!_result.stopped_eos && !_result.stopped_words) {
    _result.stopped_limited = true;
  }
  const auto t_main_end = ggml_time_us();
  _sess->get_mutex().unlock();
  if (_onComplete) {
    _onComplete();
  }
}

void LlamaCompletionWorker::OnOK() {
  auto env = Napi::AsyncWorker::Env();
  auto result = Napi::Object::New(env);
  result.Set("tokens_evaluated", Napi::Number::New(env,
                                                   _result.tokens_evaluated));
  result.Set("tokens_predicted", Napi::Number::New(Napi::AsyncWorker::Env(),
                                                   _result.tokens_predicted));
  result.Set("truncated",
             Napi::Boolean::New(env, _result.truncated));
  result.Set("context_full",
             Napi::Boolean::New(env, _result.context_full));
  result.Set("text",
             Napi::String::New(env, _result.text.c_str()));
  result.Set("stopped_eos",
             Napi::Boolean::New(env, _result.stopped_eos));
  result.Set("stopped_words",
             Napi::Boolean::New(env, _result.stopped_words));
  result.Set("stopping_word",
             Napi::String::New(env, _result.stopping_word.c_str()));
  result.Set("stopped_limited",
             Napi::Boolean::New(env, _result.stopped_limited));

  Napi::Array tool_calls = Napi::Array::New(Napi::AsyncWorker::Env());
  std::string reasoning_content = "";
  std::string content;
  if (!_stop) {
    try {
      common_chat_msg message = common_chat_parse(_result.text, static_cast<common_chat_format>(_chat_format));
      if (!message.reasoning_content.empty()) {
        reasoning_content = message.reasoning_content;
      }
      if (!message.content.empty()) {
        content = message.content;
      }
      for (size_t i = 0; i < message.tool_calls.size(); i++) {
        const auto &tc = message.tool_calls[i];
        Napi::Object tool_call = Napi::Object::New(env);
        tool_call.Set("type", "function");
        Napi::Object function = Napi::Object::New(env);
        function.Set("name", tc.name);
        function.Set("arguments", tc.arguments);
        tool_call.Set("function", function);
        if (!tc.id.empty()) {
          tool_call.Set("id", tc.id);
        }
        tool_calls.Set(i, tool_call);
      }
    } catch (const std::exception &e) {
      // console_log(env, "Error parsing tool calls: " + std::string(e.what()));
    }
  }
  if (tool_calls.Length() > 0) {
    result.Set("tool_calls", tool_calls);
  }
  if (!reasoning_content.empty()) {
    result.Set("reasoning_content", Napi::String::New(env, reasoning_content.c_str()));
  }
  if (!content.empty()) {
    result.Set("content", Napi::String::New(env, content.c_str()));
  }

  auto ctx = _sess->context();
  const auto timings_token = llama_perf_context(ctx);

  auto timingsResult = Napi::Object::New(Napi::AsyncWorker::Env());
  timingsResult.Set("prompt_n", Napi::Number::New(Napi::AsyncWorker::Env(), timings_token.n_p_eval));
  timingsResult.Set("prompt_ms", Napi::Number::New(Napi::AsyncWorker::Env(), timings_token.t_p_eval_ms));
  timingsResult.Set("prompt_per_token_ms", Napi::Number::New(Napi::AsyncWorker::Env(), timings_token.t_p_eval_ms / timings_token.n_p_eval));
  timingsResult.Set("prompt_per_second", Napi::Number::New(Napi::AsyncWorker::Env(), 1e3 / timings_token.t_p_eval_ms * timings_token.n_p_eval));
  timingsResult.Set("predicted_n", Napi::Number::New(Napi::AsyncWorker::Env(), timings_token.n_eval));
  timingsResult.Set("predicted_ms", Napi::Number::New(Napi::AsyncWorker::Env(), timings_token.t_eval_ms));
  timingsResult.Set("predicted_per_token_ms", Napi::Number::New(Napi::AsyncWorker::Env(), timings_token.t_eval_ms / timings_token.n_eval));
  timingsResult.Set("predicted_per_second", Napi::Number::New(Napi::AsyncWorker::Env(), 1e3 / timings_token.t_eval_ms * timings_token.n_eval));

  result.Set("timings", timingsResult);
  
  Napi::Promise::Deferred::Resolve(result);
}

void LlamaCompletionWorker::OnError(const Napi::Error &err) {
  Napi::Promise::Deferred::Reject(err.Value());
}
