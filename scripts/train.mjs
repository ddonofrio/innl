import { mkdirSync, writeFileSync } from 'node:fs'
import { trainModel, evaluate } from '../src/training.ts'
import { MLP, seededRandom } from '../src/mlp.ts'
import { makeExample, trainingSet, rasterize } from '../src/dataset.ts'

const iterator = trainModel()
let next = iterator.next()
while (!next.done) {
  const m = next.value.metrics
  console.log(`Epoch ${m.epoch}: train ${m.trainLoss.toFixed(4)}, validation ${m.loss.toFixed(4)}, accuracy ${(m.accuracy * 100).toFixed(2)}%, FPR ${(m.falsePositiveRate * 100).toFixed(2)}%`)
  next = iterator.next()
}
const result = next.value, model = new MLP(result.model)
// This seed is never used for training or checkpoint selection.
const test = evaluate(model, trainingSet(3000, seededRandom(700007)))
const random = seededRandom(1234567)
const lines = Array.from({ length: 2000 }, () => makeExample('line', random))
// Include a deterministic sweep of axis-aligned lines at different offsets and widths.
for (const width of [.35, .67, 1, 1.4]) for (let offset = .5; offset < 16; offset += .5) {
  for (const path of [[[.1, offset], [15.9, offset]], [[offset, .1], [offset, 15.9]]]) {
    lines.push({ pixels: rasterize([path], width), label: 0, kind: 'line' })
  }
}
const probabilities = lines.map(s => model.forward(s.pixels).at(-1)[0]).sort((a, b) => a - b)
const lineTest = { count: lines.length, ...evaluate(model, lines), maximumHeartProbability: probabilities.at(-1), p99HeartProbability: probabilities[Math.floor(probabilities.length * .99)] }
const report = { architecture: result.model.sizes, parameters: model.parameterCount, bestEpoch: result.bestEpoch, validation: result.validation, test, lineTest }
console.log(JSON.stringify(report, null, 2))
if (test.accuracy < .94 || test.recall < .92 || lineTest.falsePositiveRate > .005 || lineTest.maximumHeartProbability > .5) {
  throw new Error('Model failed the held-out acceptance checks; bundled weights were not replaced.')
}
mkdirSync(new URL('../public/', import.meta.url), { recursive: true })
writeFileSync(new URL('../public/model-v3.json', import.meta.url), JSON.stringify(result))
writeFileSync(new URL('../public/model-report.json', import.meta.url), JSON.stringify(report, null, 2) + '\n')
