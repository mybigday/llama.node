import * as path from "path";

export type ChatMessage = {
  role: string;
  content: string;
};

export type LlamaModelOptions = {
  model: string;
  chat_template?: string;
  embedding?: boolean;
  embd_normalize?: number;
  pooling_type?: "none" | "mean" | "cls" | "last" | "rank";
  n_ctx?: number;
  n_batch?: number;
  n_ubatch?: number;
  n_threads?: number;
  n_gpu_layers?: number;
  flash_attn?: boolean;
  cache_type_k?:
    | "f16"
    | "f32"
    | "q8_0"
    | "q4_0"
    | "q4_1"
    | "iq4_nl"
    | "q5_0"
    | "q5_1";
  cache_type_v?:
    | "f16"
    | "f32"
    | "q8_0"
    | "q4_0"
    | "q4_1"
    | "iq4_nl"
    | "q5_0"
    | "q5_1";
  use_mlock?: boolean;
  use_mmap?: boolean;
  vocab_only?: boolean;
  lora?: string;
  lora_scaled?: number;
  lora_list?: { path: string; scaled: number }[];
};

export type CompletionResponseFormat = {
  type: "text" | "json_object" | "json_schema";
  json_schema?: {
    strict?: boolean;
    schema: object;
  };
  schema?: object; // for json_object type
};

export type LlamaCompletionOptions = {
  messages?: ChatMessage[];
  jinja?: boolean;
  chat_template?: string;
  response_format?: CompletionResponseFormat;
  tools?: object;
  parallel_tool_calls?: boolean;
  tool_choice?: string;
  prompt?: string;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  min_p?: number;
  mirostat?: number;
  mirostat_tau?: number;
  mirostat_eta?: number;
  penalty_last_n?: number;
  penalty_repeat?: number;
  penalty_freq?: number;
  penalty_present?: number;
  typ_p?: number;
  xtc_threshold?: number;
  xtc_probability?: number;
  dry_multiplier?: number;
  dry_base?: number;
  dry_allowed_length?: number;
  dry_penalty_last_n?: number;
  n_predict?: number;
  max_length?: number;
  max_tokens?: number;
  seed?: number;
  stop?: string[];
  grammar?: string;
  grammar_lazy?: boolean;
  grammar_triggers?: { word: string; at_start: boolean }[];
  preserved_tokens?: string[];
};

export type LlamaCompletionResult = {
  text: string;
  tokens_predicted: number;
  tokens_evaluated: number;
  truncated: boolean;
  timings: {
    prompt_n: number;
    prompt_ms: number;
    prompt_per_token_ms: number;
    prompt_per_second: number;
    predicted_n: number;
    predicted_ms: number;
    predicted_per_token_ms: number;
    predicted_per_second: number;
  };
};

export type LlamaCompletionToken = {
  token: string;
};

export type TokenizeResult = {
  tokens: Int32Array;
};

export type EmbeddingResult = {
  embedding: Float32Array;
};

export interface LlamaContext {
  new (options: LlamaModelOptions): LlamaContext;
  getSystemInfo(): string;
  getModelInfo(): object;
  getFormattedChat(
    messages: ChatMessage[],
    chat_template?: string,
    params?: {
      jinja?: boolean;
      response_format?: CompletionResponseFormat;
      tools?: object;
      parallel_tool_calls?: object;
      tool_choice?: string;
    }
  ): object | string;
  completion(
    options: LlamaCompletionOptions,
    callback?: (token: LlamaCompletionToken) => void
  ): Promise<LlamaCompletionResult>;
  stopCompletion(): void;
  tokenize(text: string): Promise<TokenizeResult>;
  detokenize(tokens: number[]): Promise<string>;
  embedding(text: string): Promise<EmbeddingResult>;
  saveSession(path: string): Promise<void>;
  loadSession(path: string): Promise<void>;
  release(): Promise<void>;
  applyLoraAdapters(adapters: { path: string; scaled: number }[]): void;
  removeLoraAdapters(adapters: { path: string }[]): void;
  getLoadedLoraAdapters(): { path: string; scaled: number }[];
  // static
  loadModelInfo(path: string, skip: string[]): Promise<Object>;
}

export interface Module {
  LlamaContext: LlamaContext;
}

export type LibVariant = "default" | "vulkan" | "cuda";

const setupEnv = (variant?: string) => {
  const postfix = variant ? `-${variant}` : "";
  const binPath = path.resolve(
    __dirname,
    `../bin/${process.platform}${postfix}/${process.arch}/`
  );
  const systemPathEnv = process.env.PATH ?? process.env.Path ?? "";
  if (!systemPathEnv.includes(binPath)) {
    if (process.platform === "win32") {
      process.env.Path = `${binPath};${systemPathEnv}`;
    } else {
      process.env.PATH = `${binPath}:${systemPathEnv}`;
    }
  }
};

export const loadModule = async (variant?: LibVariant): Promise<Module> => {
  try {
    if (variant && variant !== "default") {
      setupEnv(variant);
      return (await import(
        `../bin/${process.platform}-${variant}/${process.arch}/llama-node.node`
      )) as Module;
    }
  } catch {} // ignore errors and try the common path
  setupEnv();
  return (await import(
    `../bin/${process.platform}/${process.arch}/llama-node.node`
  )) as Module;
};
