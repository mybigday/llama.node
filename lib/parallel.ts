// Parallel decoding API implementation for llama.node
import type {
  LlamaContext,
  LlamaCompletionToken,
  RerankParams,
  ParallelStatus,
  LlamaParallelCompletionOptions,
} from './binding'
import { formatMediaChat } from './utils'

export class LlamaParallelAPI {
  private context: LlamaContext
  private enabled: boolean = false
  private pendingRequests = new Map<
    number,
    {
      resolve: (value: any) => void
      reject: (reason?: any) => void
    }
  >()

  constructor(context: LlamaContext) {
    this.context = context
  }

  /**
   * Enable parallel decoding mode
   * @param config Configuration for parallel mode
   * @returns boolean indicating if successful
   */
  async enable(config?: {
    n_parallel?: number
    n_batch?: number
  }): Promise<boolean> {
    const defaultConfig = { n_parallel: 2, n_batch: 512 }
    const result = this.context.enableParallelMode({
      ...defaultConfig,
      ...config,
    })
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
  async configure(config: {
    n_parallel?: number
    n_batch?: number
  }): Promise<boolean> {
    return this.enable(config)
  }

  /**
   * Queue a completion request for parallel processing
   * @param options Completion options
   * @param onToken Optional callback for each token
   * @returns Object with requestId, promise for result, and stop function
   */
  async completion(
    options: LlamaParallelCompletionOptions,
    onToken?: (requestId: number, data: LlamaCompletionToken) => void,
  ): Promise<{
    requestId: number
    promise: Promise<any>
    stop: () => void
  }> {
    if (!this.enabled) {
      throw new Error('Parallel mode is not enabled. Call enable() first.')
    }

    const tokenCallback = onToken
      ? (error: any, result: any) => {
          if (error) {
            console.error('Token callback error:', error)
            // Handle completion error
            const pendingReq = this.pendingRequests.get(result?.requestId)
            if (pendingReq) {
              pendingReq.reject(error)
              this.pendingRequests.delete(result?.requestId)
            }
            return
          }
          // Check if this is a token callback or final result
          if (result) {
            if (result.token !== undefined) {
              // This is a token callback
              onToken(result.requestId, result)
            } else if (
              result.text !== undefined ||
              result.content !== undefined
            ) {
              // This is the final result
              const pendingReq = this.pendingRequests.get(result.requestId)
              if (pendingReq) {
                pendingReq.resolve(result)
                this.pendingRequests.delete(result.requestId)
              }
            }
          }
        }
      : undefined

    const { messages, media_paths = options.media_paths } = formatMediaChat(
      options.messages,
    )
    // Queue the completion immediately (this is synchronous!)
    const { requestId } = this.context.queueCompletion(
      {
        ...options,
        messages,
        media_paths: media_paths,
      },
      tokenCallback ||
        ((error, result) => {
          if (error) {
            const pendingReq = this.pendingRequests.get(result?.requestId)
            if (pendingReq) {
              pendingReq.reject(error)
              this.pendingRequests.delete(result?.requestId)
            }
          } else if (
            result &&
            (result.text !== undefined || result.content !== undefined)
          ) {
            // Final result for non-streaming
            const pendingReq = this.pendingRequests.get(result.requestId)
            if (pendingReq) {
              pendingReq.resolve(result)
              this.pendingRequests.delete(result.requestId)
            }
          }
        }),
    )

    // Create promise for final result
    const promise = new Promise((resolveResult, rejectResult) => {
      this.pendingRequests.set(requestId, {
        resolve: resolveResult,
        reject: rejectResult,
      })
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

    // Return immediately without wrapping in a Promise
    return {
      requestId,
      promise,
      stop,
    }
  }

  /**
   * Queue an embedding request for parallel processing
   * @param text Text to embed
   * @param params Optional embedding parameters
   * @returns Object with requestId and promise for result
   */
  async embedding(
    text: string,
    params?: { embd_normalize?: number },
  ): Promise<{
    requestId: number
    promise: Promise<{ embedding: number[] }>
  }> {
    if (!this.enabled) {
      throw new Error('Parallel mode is not enabled. Call enable() first.')
    }

    // Create promise for result
    let resolveResult: (value: any) => void
    let rejectResult: (reason?: any) => void

    const promise = new Promise<{ embedding: number[] }>((res, rej) => {
      resolveResult = res
      rejectResult = rej
    })

    // Queue the embedding immediately (this is synchronous!)
    const { requestId } = this.context.queueEmbedding(
      text,
      params,
      (error, result) => {
        if (error) {
          rejectResult(error)
        } else {
          resolveResult(result)
        }
      },
    )

    // Return immediately without wrapping in a Promise
    return {
      requestId,
      promise,
    }
  }

  /**
   * Queue a rerank request for parallel processing
   * @param query Query text
   * @param documents Documents to rank
   * @param params Optional rerank parameters
   * @returns Object with requestId and promise for results
   */
  async rerank(
    query: string,
    documents: string[],
    params?: RerankParams,
  ): Promise<{
    requestId: number
    promise: Promise<Array<{ score: number; index: number; document: string }>>
  }> {
    if (!this.enabled) {
      throw new Error('Parallel mode is not enabled. Call enable() first.')
    }

    // Create promise for result
    let resolveResult: (value: any) => void
    let rejectResult: (reason?: any) => void

    const promise = new Promise<
      Array<{ score: number; index: number; document: string }>
    >((res, rej) => {
      resolveResult = res
      rejectResult = rej
    })

    // Queue the rerank immediately (this is synchronous!)
    const { requestId } = this.context.queueRerank(
      query,
      documents,
      params,
      (error, result) => {
        if (error) {
          rejectResult(error)
        } else {
          // Add document text to results and sort by score
          const enrichedResults = result.results
            .map((r: any) => ({
              ...r,
              document: documents[r.index],
            }))
            .sort((a: any, b: any) => b.score - a.score)
          resolveResult(enrichedResults)
        }
      },
    )

    // Return immediately without wrapping in a Promise
    return {
      requestId,
      promise,
    }
  }

  /**
   * Check if parallel mode is enabled
   */
  isEnabled(): boolean {
    return this.enabled
  }

  /**
   * Get current parallel processing status (one-time snapshot)
   * @returns Current parallel status
   */
  getStatus(): ParallelStatus {
    if (!this.enabled) {
      throw new Error('Parallel mode is not enabled. Call enable() first.')
    }
    return this.context.getParallelStatus()
  }

  /**
   * Subscribe to parallel processing status changes
   * @param callback Called whenever parallel status changes
   * @returns Object with remove() method to unsubscribe
   */
  subscribeToStatus(
    callback: (status: ParallelStatus) => void,
  ): { remove: () => void } {
    if (!this.enabled) {
      throw new Error('Parallel mode is not enabled. Call enable() first.')
    }

    const { subscriberId } = this.context.subscribeParallelStatus(callback)

    return {
      remove: () => {
        this.context.unsubscribeParallelStatus(subscriberId)
      },
    }
  }
}
