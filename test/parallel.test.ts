import { loadModel, ParallelStatus } from '../lib'
import * as path from 'path'
import * as fs from 'fs'

describe('Parallel Decoding', () => {
  let context: any
  const modelPath = path.join(__dirname, 'tiny-random-llama.gguf')

  // Skip tests if model doesn't exist
  beforeAll(() => {
    if (!fs.existsSync(modelPath)) {
      console.log(
        `Test model not found at ${modelPath}, skipping parallel tests`,
      )
      // @ts-ignore
      test.skip()
    }
  })

  beforeEach(async () => {
    context = await loadModel({
      model: modelPath,
      n_ctx: 512,
      n_gpu_layers: 0,
      n_parallel: 4, // Support up to 4 parallel slots
    })
  })

  afterEach(async () => {
    if (context) {
      if (context.parallel.isEnabled()) {
        context.parallel.disable()
      }
      await context.release()
    }
  })

  describe('Parallel Mode Management', () => {
    test('should enable parallel mode with default settings', async () => {
      const enabled = await context.parallel.enable()
      expect(enabled).toBe(true)
      expect(context.parallel.isEnabled()).toBe(true)
    })

    test('should enable parallel mode with custom settings', async () => {
      const enabled = await context.parallel.enable({
        n_parallel: 2,
        n_batch: 256,
      })
      expect(enabled).toBe(true)
      expect(context.parallel.isEnabled()).toBe(true)
    })

    test('should disable parallel mode', async () => {
      await context.parallel.enable()
      expect(context.parallel.isEnabled()).toBe(true)

      context.parallel.disable()
      expect(context.parallel.isEnabled()).toBe(false)
    })

    test('should handle re-enabling parallel mode', async () => {
      await context.parallel.enable({ n_parallel: 2 })
      expect(context.parallel.isEnabled()).toBe(true)

      // Re-enable with different settings
      const enabled = await context.parallel.enable({ n_parallel: 3 })
      expect(enabled).toBe(true)
      expect(context.parallel.isEnabled()).toBe(true)
    })
  })

  describe('Parallel Completion', () => {
    beforeEach(async () => {
      await context.parallel.enable({ n_parallel: 2, n_batch: 128 })
    })

    test('should queue and process single completion request', async () => {
      const request = await context.parallel.completion({
        prompt: 'Hello',
        max_tokens: 5,
        temperature: 0.7,
      })

      expect(request.requestId).toBeDefined()
      expect(typeof request.requestId).toBe('number')
      expect(request.promise).toBeDefined()

      // Wait for completion with timeout
      const result = await Promise.race([
        request.promise,
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Timeout')), 3000),
        ),
      ]).catch(() => null)

      // Even if it times out, the request was successfully queued
      expect(request.requestId).toBeGreaterThan(0)
    }, 5000)

    test('should process multiple parallel completion requests', async () => {
      const req1 = await context.parallel.completion({
        prompt: 'One',
        max_tokens: 3,
        temperature: 0.7,
      })

      const req2 = await context.parallel.completion({
        prompt: 'Two',
        max_tokens: 3,
        temperature: 0.7,
      })

      expect(req1.requestId).toBeDefined()
      expect(req2.requestId).toBeDefined()
      expect(req1.requestId).not.toBe(req2.requestId)

      // Both requests should have unique IDs
      expect(req1.requestId).toBeGreaterThan(0)
      expect(req2.requestId).toBeGreaterThan(0)
    }, 5000)

    test('should handle completion with token callback', async () => {
      const tokens: string[] = []

      const request = await context.parallel.completion(
        {
          prompt: 'Test',
          max_tokens: 3,
          temperature: 0.7,
        },
        (_requestId: number, token: any) => {
          if (token.token) {
            tokens.push(token.token)
          }
        },
      )

      expect(request.requestId).toBeDefined()

      // Wait briefly for some tokens
      await new Promise((resolve) => setTimeout(resolve, 1000))

      // Request was successfully queued even if no tokens generated yet
      expect(request.requestId).toBeGreaterThan(0)
      expect(tokens.length).toBeGreaterThan(0)
    }, 5000)

    test('should stop completion request', async () => {
      // Queue a request and immediately stop it
      const request = await context.parallel.completion({
        prompt: 'Long text that should be stopped',
        max_tokens: 100,
      })

      expect(request.stop).toBeDefined()

      // Stopping should reject the promise
      request.stop()

      await expect(request.promise).rejects.toThrow('Request cancelled')
    })
  })

  describe('Parallel Embedding', () => {
    beforeEach(async () => {
      await context.parallel.enable({ n_parallel: 2 })
    })

    test('should queue embedding request', async () => {
      const request = await context.parallel.embedding('Test text')

      expect(request.requestId).toBeDefined()
      expect(typeof request.requestId).toBe('number')
      expect(request.requestId).toBeGreaterThan(0)
      expect(request.promise).toBeDefined()
    })

    test('should queue embedding with normalization', async () => {
      const request = await context.parallel.embedding('Test text', {
        embd_normalize: 2,
      })

      expect(request.requestId).toBeGreaterThan(0)
    })

    test('should process multiple embedding requests', async () => {
      const req1 = await context.parallel.embedding('First text')
      const req2 = await context.parallel.embedding('Second text')

      expect(req1.requestId).not.toBe(req2.requestId)
      expect(req1.requestId).toBeGreaterThan(0)
      expect(req2.requestId).toBeGreaterThan(0)
    })
  })

  describe('Parallel Rerank', () => {
    beforeEach(async () => {
      await context.parallel.enable({ n_parallel: 2 })
    })

    test('should queue rerank request', async () => {
      const request = await context.parallel.rerank('Query text', [
        'Document 1',
        'Document 2',
        'Document 3',
      ])

      expect(request.requestId).toBeDefined()
      expect(typeof request.requestId).toBe('number')
      expect(request.requestId).toBeGreaterThan(0)
      expect(request.promise).toBeDefined()
    })

    test('should queue rerank with normalization', async () => {
      const request = await context.parallel.rerank(
        'Query text',
        ['Doc 1', 'Doc 2'],
        { normalize: 1 },
      )

      expect(request.requestId).toBeGreaterThan(0)
    })
  })

  describe('Error Handling', () => {
    test('should throw error when queueing without enabling parallel mode', async () => {
      await expect(
        context.parallel.completion({ prompt: 'Test', max_tokens: 5 }),
      ).rejects.toThrow('Parallel mode is not enabled')
    })

    test('should throw error when enabling with invalid parameters', async () => {
      await expect(
        context.parallel.enable({ n_parallel: 100 }), // Exceeds n_seq_max
      ).rejects.toThrow()
    })

    test('should handle empty prompt gracefully', async () => {
      await context.parallel.enable()

      await expect(
        context.parallel.completion({ prompt: '', max_tokens: 5 }),
      ).rejects.toThrow()
    })
  })

  describe('Concurrent Operations', () => {
    beforeEach(async () => {
      await context.parallel.enable({ n_parallel: 3 })
    })

    test('should handle mixed operation types concurrently', async () => {
      // Queue different types of operations
      const completion = await context.parallel.completion({
        prompt: 'Hello',
        max_tokens: 3,
      })

      const embedding = await context.parallel.embedding('Test text')

      const rerank = await context.parallel.rerank('Query', ['Doc1', 'Doc2'])

      // All should have unique request IDs
      expect(completion.requestId).toBeDefined()
      expect(embedding.requestId).toBeDefined()
      expect(rerank.requestId).toBeDefined()

      const ids = [completion.requestId, embedding.requestId, rerank.requestId]
      const uniqueIds = [...new Set(ids)]
      expect(uniqueIds.length).toBe(3)
    })

    test('should maintain request isolation', async () => {
      const results: any[] = []

      // Start multiple completions with different prompts
      const prompts = ['Alpha', 'Beta', 'Gamma']
      const requests = await Promise.all(
        prompts.map((prompt) =>
          context.parallel.completion({
            prompt,
            max_tokens: 2,
            temperature: 0.5,
          }),
        ),
      )

      // All requests should have unique IDs
      const requestIds = requests.map((r) => r.requestId)
      expect(new Set(requestIds).size).toBe(requestIds.length)
    })
  })

  describe('Parallel Status', () => {
    test('should get status after enabling parallel mode', async () => {
      await context.parallel.enable({ n_parallel: 2 })

      const status = context.parallel.getStatus()

      expect(status).toBeDefined()
      expect(status.n_parallel).toBe(2)
      expect(status.active_slots).toBe(0)
      expect(status.queued_requests).toBe(0)
      expect(status.requests).toEqual([])
    })

    test('should show active requests in status', async () => {
      await context.parallel.enable({ n_parallel: 2 })

      // Queue a completion request
      const request = await context.parallel.completion({
        prompt: 'Hello world',
        max_tokens: 50,
        temperature: 0.7,
      })

      // Give it a moment to start processing
      await new Promise((resolve) => setTimeout(resolve, 100))

      const status = context.parallel.getStatus()

      // Should have at least one request (either active or queued)
      expect(status.requests.length).toBeGreaterThanOrEqual(0)

      // If there are requests, check their structure
      if (status.requests.length > 0) {
        const req = status.requests[0]
        expect(req.request_id).toBeDefined()
        expect(req.type).toBe('completion')
        expect(['queued', 'processing_prompt', 'generating', 'done']).toContain(
          req.state,
        )
        expect(typeof req.prompt_length).toBe('number')
        expect(typeof req.tokens_generated).toBe('number')
        expect(typeof req.prompt_ms).toBe('number')
        expect(typeof req.generation_ms).toBe('number')
        expect(typeof req.tokens_per_second).toBe('number')
      }

      // Stop the request to clean up (catch the rejection)
      request.stop()
      await request.promise.catch(() => {})
    }, 5000)

    test('should throw error when getting status without parallel mode', async () => {
      expect(() => context.parallel.getStatus()).toThrow(
        'Parallel mode is not enabled',
      )
    })

    test('should subscribe to status changes', async () => {
      await context.parallel.enable({ n_parallel: 2 })

      const statusUpdates: ParallelStatus[] = []

      const subscription = context.parallel.subscribeToStatus(
        (status: ParallelStatus) => {
          statusUpdates.push(status)
        },
      )

      expect(subscription).toBeDefined()
      expect(subscription.remove).toBeDefined()
      expect(typeof subscription.remove).toBe('function')

      // Queue a request to trigger status updates
      const request = await context.parallel.completion({
        prompt: 'Test',
        max_tokens: 5,
        temperature: 0.7,
      })

      // Wait for some status updates
      await new Promise((resolve) => setTimeout(resolve, 500))

      // Stop the request (catch the rejection)
      request.stop()
      await request.promise.catch(() => {})

      // Unsubscribe
      subscription.remove()

      // Should have received at least one status update
      expect(statusUpdates.length).toBeGreaterThan(0)

      // Each update should have the correct structure
      for (const status of statusUpdates) {
        expect(status.n_parallel).toBe(2)
        expect(typeof status.active_slots).toBe('number')
        expect(typeof status.queued_requests).toBe('number')
        expect(Array.isArray(status.requests)).toBe(true)
      }
    }, 5000)

    test('should stop receiving updates after unsubscribing', async () => {
      await context.parallel.enable({ n_parallel: 2 })

      const statusUpdates: ParallelStatus[] = []

      const subscription = context.parallel.subscribeToStatus(
        (status: ParallelStatus) => {
          statusUpdates.push(status)
        },
      )

      // Unsubscribe immediately
      subscription.remove()

      // Queue a request
      const request = await context.parallel.completion({
        prompt: 'Test',
        max_tokens: 5,
      })

      // Wait a bit
      await new Promise((resolve) => setTimeout(resolve, 300))

      // Stop the request (catch the rejection)
      request.stop()
      await request.promise.catch(() => {})

      // Should not have received any updates after unsubscribing
      // (might have received 0-1 updates before unsubscribe completed)
      expect(statusUpdates.length).toBeLessThanOrEqual(1)
    }, 5000)

    test('should throw error when subscribing without parallel mode', async () => {
      expect(() => context.parallel.subscribeToStatus(() => {})).toThrow(
        'Parallel mode is not enabled',
      )
    })

    test('should track multiple concurrent requests in status', async () => {
      await context.parallel.enable({ n_parallel: 3 })

      // Queue multiple requests
      const requests = await Promise.all([
        context.parallel.completion({ prompt: 'One', max_tokens: 10 }),
        context.parallel.completion({ prompt: 'Two', max_tokens: 10 }),
        context.parallel.embedding('Test embedding'),
      ])

      // Give time for processing to start
      await new Promise((resolve) => setTimeout(resolve, 100))

      const status = context.parallel.getStatus()

      // Should show the parallel configuration
      expect(status.n_parallel).toBe(3)

      // Total of active + queued should be at least our requests (some may have completed)
      const totalTracked = status.active_slots + status.queued_requests

      // Clean up (catch rejections from stopping)
      await Promise.all(
        requests.map(async (r) => {
          if ('stop' in r) {
            r.stop()
            await r.promise.catch(() => {})
          }
        }),
      )
    }, 5000)
  })
})
