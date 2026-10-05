import path from 'path'
import { loadModel } from '../lib'

const decisionModel = path.resolve(__dirname, './Julia-1-Q8_0.gguf')
const textModel = path.resolve(__dirname, './tiny-random-llama.gguf')

const request = {
  state: { utterance: 'one medium fries please', cart: [] },
  questions: {
    is_ordering: { type: 'noul', instructions: 'The customer is ordering' },
    product: {
      type: 'choice',
      instructions: 'Which product is ordered?',
      criteria: { 'fries-m': 'Fries M', burger: 'Burger', none: null },
    },
    mood: {
      type: 'score',
      instructions: 'How impatient is the customer?',
      criteria: ['calm', 'neutral', 'impatient'],
    },
  },
} as const

const sum = (values: Record<string, number>) =>
  Object.values(values).reduce((a, b) => a + b, 0)

describe('Decision', () => {
  let context: Awaited<ReturnType<typeof loadModel>>

  beforeAll(async () => {
    context = await loadModel({
      model: decisionModel,
      n_ctx: 2048,
      n_parallel: 2,
    })
  })

  afterAll(async () => {
    if (context.parallel.isEnabled()) context.parallel.disable()
    await context.release()
  })

  test('model info', () => {
    expect(context.getModelInfo().decision).toEqual({
      type: 'laya',
      nOptionsMax: expect.any(Number),
      imageInput: false,
      textGeneration: false,
    })
  })

  test('decide', async () => {
    const result = await context.decide(request)
    expect(result.usage.input_tokens).toBeGreaterThan(0)
    expect(result.usage.output_tokens).toBe(0)

    const { answers } = result
    // the keys of a choice question narrow its answer
    const choice: 'fries-m' | 'burger' | 'none' = answers.product.choice
    expect(choice).toBe('fries-m')
    expect(Object.keys(answers.product.probabilities).sort()).toEqual([
      'burger',
      'fries-m',
      'none',
    ])
    expect(sum(answers.product.probabilities)).toBeCloseTo(1, 5)
    expect(answers.product.confidence).toBeGreaterThan(0)
    expect(answers.product.confidence).toBeLessThanOrEqual(1)

    expect(answers.is_ordering.type).toBe('noul')
    expect(answers.is_ordering.noul).toBeGreaterThan(0.5)

    expect(answers.mood.type).toBe('score')
    expect(answers.mood.legend).toEqual({
      0: 'calm',
      1: 'neutral',
      2: 'impatient',
    })
    expect(sum(answers.mood.probabilities)).toBeCloseTo(1, 5)
    expect(answers.mood.score).toBeGreaterThanOrEqual(0)
    expect(answers.mood.score).toBeLessThanOrEqual(2)

    // each answer is read from one forward pass, the result is deterministic
    expect((await context.decide(request)).answers).toEqual(answers)
  })

  test('decide with option keys that are not ASCII', async () => {
    const { answers } = await context.decide({
      state: '我要一份中薯',
      questions: {
        product: {
          type: 'choice',
          instructions: '顧客點了什麼？',
          criteria: { 中薯: '中份薯條', 漢堡: '漢堡', 無: null },
        },
      },
    })
    expect(Object.keys(answers.product.probabilities).sort()).toEqual(
      ['中薯', '漢堡', '無'].sort(),
    )
    expect(answers.product.choice).toBe('中薯')
  })

  test('decide rejects an invalid request', async () => {
    await expect(
      context.decide({
        state: 'x',
        questions: { q: { type: 'bogus', instructions: 'x' } } as any,
      }),
    ).rejects.toThrow(/type/)
  })

  test('completion is refused for a decision-only model', async () => {
    await expect(
      (async () => context.completion({ prompt: 'Hello', n_predict: 1 }))(),
    ).rejects.toThrow('This model only answers decisions')
  })

  test('parallel decide', async () => {
    const expected = await context.decide(request)

    await context.parallel.enable({ n_parallel: 2 })
    try {
      // the context's own sequence belongs to the slots now
      await expect(context.decide(request)).rejects.toThrow(
        'use parallel.decide()',
      )

      const a = await context.parallel.decide(request)
      const b = await context.parallel.decide({
        ...request,
        state: 'I just want to look at the menu',
      })
      expect(typeof a.requestId).toBe('number')
      expect(a.requestId).not.toBe(b.requestId)

      const [resultA, resultB] = await Promise.all([a.promise, b.promise])
      expect(resultA.answers).toEqual(expected.answers)
      expect(resultB.answers.product.choice).toBe('none')

      // an invalid request is refused before it is queued
      await expect(
        context.parallel.decide({
          state: 'x',
          questions: { q: { type: 'bogus', instructions: 'x' } } as any,
        }),
      ).rejects.toThrow(/type/)

      const stopped = await context.parallel.decide(request)
      stopped.stop()
      await expect(stopped.promise).rejects.toThrow('cancelled')
    } finally {
      context.parallel.disable()
    }
  })
})

test('decide rejects a model that is not a decision model', async () => {
  const context = await loadModel({ model: textModel })
  try {
    expect(context.getModelInfo().decision).toBeUndefined()
    await expect(
      context.decide({
        state: 'x',
        questions: { q: { type: 'noul', instructions: 'x' } },
      }),
    ).rejects.toThrow('not a decision model')
  } finally {
    await context.release()
  }
})
