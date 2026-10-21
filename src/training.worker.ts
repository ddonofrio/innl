import { trainModel, type TrainingMessage } from './training.ts'
import type { SerializedModel } from './mlp.ts'

self.onmessage = (event: MessageEvent<{ model: SerializedModel; seed: number }>) => {
  try {
    const iterator = trainModel(event.data.model, event.data.seed)
    let next = iterator.next()
    while (!next.done) { self.postMessage(next.value); next = iterator.next() }
    self.postMessage({ type: 'done', result: next.value } satisfies TrainingMessage)
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : 'Training failed.' } satisfies TrainingMessage)
  }
}
