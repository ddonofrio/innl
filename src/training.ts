import { MLP, seededRandom, type SerializedModel } from './mlp.ts'
import { trainingSet, type Example } from './dataset.ts'

export type Evaluation = { loss: number; accuracy: number; falsePositiveRate: number; recall: number }
export type Metrics = Evaluation & { epoch: number; trainLoss: number; trainAccuracy: number }
export type TrainingResult = { model: SerializedModel; history: Metrics[]; bestEpoch: number; validation: Evaluation }
export type TrainingMessage =
  | { type: 'progress'; metrics: Metrics; model: SerializedModel }
  | { type: 'done'; result: TrainingResult }
  | { type: 'error'; message: string }

export function evaluate(model: MLP, samples: Example[]): Evaluation {
  let loss = 0, correct = 0, falsePositive = 0, truePositive = 0, positives = 0
  for (const { pixels, label } of samples) {
    const p = model.forward(pixels).at(-1)![0]
    loss -= label * Math.log(Math.max(1e-7, p)) + (1 - label) * Math.log(Math.max(1e-7, 1 - p))
    correct += Number((p >= .5) === (label === 1))
    positives += label
    if (p >= .5) { if (label) truePositive++; else falsePositive++ }
  }
  return { loss: loss / samples.length, accuracy: correct / samples.length, falsePositiveRate: falsePositive / Math.max(1, samples.length - positives), recall: truePositive / Math.max(1, positives) }
}

export function* trainModel(saved?: SerializedModel, seed = 8427, maxEpochs = 60): Generator<Extract<TrainingMessage, { type: 'progress' }>, TrainingResult> {
  const random = seededRandom(seed), model = new MLP(saved, random)
  // Validation is fixed and independent of the fresh examples generated each epoch.
  const validation = trainingSet(800, seededRandom(92021))
  let best = model.serialize(), bestMetrics = evaluate(model, validation), bestEpoch = 0, stale = 0
  const history: Metrics[] = []
  for (let epoch = 1; epoch <= maxEpochs; epoch++) {
    const samples = trainingSet(2048, random)
    // Unbiased Fisher-Yates shuffle.
    for (let i = samples.length - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [samples[i], samples[j]] = [samples[j], samples[i]] }
    let loss = 0, correct = 0
    const rate = .0012 * (.35 + .65 * (1 - epoch / maxEpochs))
    for (let at = 0; at < samples.length; at += 32) {
      const batch = samples.slice(at, at + 32)
      const result = model.trainBatch(batch.map(s => s.pixels), batch.map(s => s.label), rate)
      loss += result.loss * batch.length; correct += result.correct
    }
    const metrics = { ...evaluate(model, validation), epoch, trainLoss: loss / samples.length, trainAccuracy: correct / samples.length }
    if (!Number.isFinite(metrics.loss)) throw new Error('Training produced a non-finite loss. The last saved model is unchanged.')
    history.push(metrics)
    if (metrics.loss < bestMetrics.loss - .0001) {
      best = model.serialize(); bestMetrics = metrics; bestEpoch = epoch; stale = 0
    } else stale++
    yield { type: 'progress', metrics, model: model.serialize() }
    if (epoch >= 20 && stale >= 12) break
  }
  return { model: best, history, bestEpoch, validation: bestMetrics }
}
