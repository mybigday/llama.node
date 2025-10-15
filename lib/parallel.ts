// Parallel decoding API implementation for llama.node
import type { LlamaContext, LlamaCompletionOptions, LlamaCompletionToken, RerankParams } from './binding'

export class LlamaParallelAPI {
  private context: LlamaContext
  private enabled: boolean = false
  private pendingRequests = new Map<number, {
    resolve: (value: any) => void
    reject: (reason?: any) => void
  }>()

  constructor(context: LlamaContext) {
    this.context = context
  }

  /**
   * Enable parallel decoding mode
   * @param config Configuration for parallel mode
   * @returns boolean indicating if successful
   */
  async enable(config?: { n_parallel?: number, n_batch?: number }): Promise<boolean> {
    const defaultConfig = { n_parallel: 2, n_batch: 512 }
    const result = this.context.enableParallelMode({ ...defaultConfig, ...config })
    this.enabled = result
    return result
  }

  /**
   * Disable parallel decoding mode
   */
  disable(): void {
    this.context.disableParallelMode()
    this.enabled = false
  }

  /**
   * Configure parallel decoding mode (enables if not already enabled)
   * @param config Configuration for parallel mode
   * @returns boolean indicating if successful
   */
  async configure(config: { n_parallel?: number, n_batch?: number }): Promise<boolean> {
    return this.enable(config)
  }

  /**
   * Queue a completion request for parallel processing
   * @param options Completion options
   * @param onToken Optional callback for each token
   * @returns Promise with requestId, promise for result, and stop function
   */
  async completion(
    options: LlamaCompletionOptions,
    onToken?: (requestId: number, data: LlamaCompletionToken) => void
  ): Promise<{
    requestId: number
    promise: Promise<any>
    stop: () => void
  }> {
    if (!this.enabled) {
      throw new Error('Parallel mode is not enabled. Call enable() first.')
    }

    return new Promise((resolve, reject) => {
      try {
        const tokenCallback = onToken ? (error: any, result: any) => {
          if (error) {
            console.error('Token callback error:', error)
            return
          }
          if (result.token || result.content) {
            onToken(result.requestId, result)
          } else if (result.text) {
            // Final result
            const pendingReq = this.pendingRequests.get(result.requestId)
            if (pendingReq) {
              pendingReq.resolve(result)
              this.pendingRequests.delete(result.requestId)
            }
          }
        } : undefined

        const { requestId } = this.context.queueCompletion(options, tokenCallback)

        // Create promise for final result
        const promise = new Promise((resolveResult, rejectResult) => {
          this.pendingRequests.set(requestId, {
            resolve: resolveResult,
            reject: rejectResult
          })

          // Set up completion callback if no token callback
          if (!tokenCallback) {
            // For non-streaming, we need to poll or rely on internal callbacks
            // This is a simplified approach - real implementation would use proper event handling
            setTimeout(() => {
              const pendingReq = this.pendingRequests.get(requestId)
              if (pendingReq) {
                pendingReq.reject(new Error('Request timeout'))
                this.pendingRequests.delete(requestId)
              }
            }, 300000) // 5 minute timeout
          }
        })

        // Create stop function
        const stop = () => {
          this.context.cancelRequest(requestId)
          const pendingReq = this.pendingRequests.get(requestId)
          if (pendingReq) {
            pendingReq.reject(new Error('Request cancelled'))
            this.pendingRequests.delete(requestId)
          }
        }

        resolve({
          requestId,
          promise,
          stop
        })
      } catch (error) {
        reject(error)
      }
    })
  }

  /**
   * Queue an embedding request for parallel processing
   * @param text Text to embed
   * @param params Optional embedding parameters
   * @returns Promise with requestId and promise for result
   */
  async embedding(
    text: string,
    params?: { embd_normalize?: number }
  ): Promise<{
    requestId: number
    promise: Promise<{ embedding: number[] }>
  }> {
    if (!this.enabled) {
      throw new Error('Parallel mode is not enabled. Call enable() first.')
    }

    return new Promise((resolve, reject) => {
      try {
        // Create promise for result
        let resolveResult: (value: any) => void
        let rejectResult: (reason?: any) => void

        const promise = new Promise<{ embedding: number[] }>((res, rej) => {
          resolveResult = res
          rejectResult = rej
        })

        const { requestId } = this.context.queueEmbedding(
          text,
          params,
          (error, result) => {
            if (error) {
              rejectResult(error)
            } else {
              resolveResult(result)
            }
          }
        )

        resolve({
          requestId,
          promise
        })
      } catch (error) {
        reject(error)
      }
    })
  }

  /**
   * Queue a rerank request for parallel processing
   * @param query Query text
   * @param documents Documents to rank
   * @param params Optional rerank parameters
   * @returns Promise with requestId and promise for results
   */
  async rerank(
    query: string,
    documents: string[],
    params?: RerankParams
  ): Promise<{
    requestId: number
    promise: Promise<Array<{ score: number, index: number, document: string }>>
  }> {
    if (!this.enabled) {
      throw new Error('Parallel mode is not enabled. Call enable() first.')
    }

    return new Promise((resolve, reject) => {
      try {
        // Create promise for result
        let resolveResult: (value: any) => void
        let rejectResult: (reason?: any) => void

        const promise = new Promise<Array<{ score: number, index: number, document: string }>>((res, rej) => {
          resolveResult = res
          rejectResult = rej
        })

        const { requestId } = this.context.queueRerank(
          query,
          documents,
          params,
          (error, result) => {
            if (error) {
              rejectResult(error)
            } else {
              // Add document text to results and sort by score
              const enrichedResults = result.results.map((r: any) => ({
                ...r,
                document: documents[r.index]
              })).sort((a: any, b: any) => b.score - a.score)
              resolveResult(enrichedResults)
            }
          }
        )

        resolve({
          requestId,
          promise
        })
      } catch (error) {
        reject(error)
      }
    })
  }

  /**
   * Check if parallel mode is enabled
   */
  isEnabled(): boolean {
    return this.enabled
  }
}