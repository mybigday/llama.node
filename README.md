# llama.node

Node binding of [llama.cpp](https://github.com/ggerganov/llama.cpp).

[llama.cpp](https://github.com/ggerganov/llama.cpp): Inference of [LLaMA](https://arxiv.org/abs/2302.13971) model in pure C/C++

## Installation

```sh
npm install llama.node
```

## Usage

```js
import { loadModel } from 'llama.node'

// Initial a Llama context with the model (may take a while)
const context = loadModel({
  model: 'path/to/gguf/model',
  use_mlock: true,
  n_ctx: 2048,
  n_gpu_layers: 1, // > 0: enable Metal on iOS
  // embedding: true, // use embedding
})

// Do completion
const { text, timings } = await context.completion(
  {
    prompt: 'This is a conversation between user and llama, a friendly chatbot. respond in simple markdown.\n\nUser: Hello!\nLlama:',
    n_predict: 100,
    stop: ['</s>', 'Llama:', 'User:'],
    // n_threads: 4,
  },
  (data) => {
    // This is a partial completion callback
    const { token } = data
  },
)
console.log('Result:', text)
```

## License

MIT

---

<p align="center">
  <a href="https://bricks.tools">
    <img width="90px" src="https://avatars.githubusercontent.com/u/17320237?s=200&v=4">
  </a>
  <p align="center">
    Built and maintained by <a href="https://bricks.tools">BRICKS</a>.
  </p>
</p>
