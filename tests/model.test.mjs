import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { MLP, seededRandom, validModel, ARCHITECTURE } from '../src/mlp.ts'
import { makeExample, rasterize, trainingSet } from '../src/dataset.ts'
import { evaluate, trainModel } from '../src/training.ts'

const bundled = JSON.parse(readFileSync(new URL('../public/model-v3.json', import.meta.url), 'utf8'))

test('bundled checkpoint validates and round-trips exact activations', () => {
  assert.ok(validModel(bundled.model))
  const model = new MLP(bundled.model), clone = new MLP(model.serialize())
  assert.equal(model.parameterCount, 19201)
  const input = makeExample('heart', seededRandom(33)).pixels
  assert.deepEqual(model.forward(input), clone.forward(input))
  assert.deepEqual(model.forward(input).map(layer => layer.length), [...ARCHITECTURE])
  assert.equal(validModel({ ...bundled.model, version: 1 }), false)
  const corrupt = structuredClone(bundled.model); corrupt.weights[0][0] = NaN
  assert.equal(validModel(corrupt), false)
})

test('backpropagation matches numerical derivatives in every layer', () => {
  const model = new MLP(undefined, seededRandom(123))
  // ReLU is not differentiable at zero. Keep this finite-difference fixture
  // strictly inside the active region, including for blank input examples.
  for (const layer of model.layers) {
    layer.biases.fill(.2)
    for (let i = 0; i < layer.weights.length; i++) layer.weights[i] = Math.abs(layer.weights[i]) * .2
  }
  const samples = trainingSet(6, seededRandom(321)), xs = samples.map(s => s.pixels), ys = samples.map(s => s.label)
  const gradients = model.gradients(xs, ys).gradients
  for (let l = 0; l < model.layers.length; l++) for (const key of ['weights', 'biases']) {
    const p = model.layers[l][key], g = gradients[l][key]
    const probes = Array.from(g, (gradient, index) => ({ gradient, index })).sort((a, b) => Math.abs(b.gradient) - Math.abs(a.gradient)).slice(0, 4)
    for (const { index, gradient } of probes) {
      const initial = p[index], h = .0005
      p[index] = initial + h; const upper = evaluate(model, samples).loss
      p[index] = initial - h; const lower = evaluate(model, samples).loss
      p[index] = initial
      const numerical = (upper - lower) / (2 * h)
      assert.ok(Math.abs(numerical - gradient) < .002 + Math.abs(gradient) * .04, `layer ${l} ${key}[${index}]: ${gradient} vs ${numerical}`)
    }
  }
})

test('inactive ReLU units block gradients', () => {
  const model = new MLP(undefined, seededRandom(123))
  model.layers[0].biases.fill(-100)
  const input = makeExample('heart', seededRandom(123)).pixels
  const result = model.gradients([input], [1])
  assert.ok(model.forward(input)[1].every(value => value === 0))
  assert.ok(result.gradients[0].weights.every(value => value === 0))
  assert.ok(result.gradients[0].biases.every(value => value === 0))
})

test('layer ablation zeroes the selected hidden layer and changes downstream inference', () => {
  const model = new MLP(bundled.model), input = makeExample('heart', seededRandom(91)).pixels
  const normal = model.forward(input), ablated = model.forward(input, new Set([2]))
  assert.ok(normal[2].some(value => value > 0))
  assert.ok(ablated[2].every(value => value === 0))
  assert.notEqual(ablated.at(-1)[0], normal.at(-1)[0])
})

test('manual neuron interventions propagate forward and respect layer ablation', () => {
  const model = new MLP(bundled.model), input = makeExample('heart', seededRandom(191)).pixels
  const normal = model.forward(input), overrides = new Map([['2:0', normal[2][0] + 3]])
  const edited = model.forward(input, new Set(), overrides)
  assert.equal(edited[2][0], normal[2][0] + 3)
  assert.notDeepEqual(edited.slice(3), normal.slice(3))
  assert.equal(model.forward(input, new Set([2]), overrides)[2][0], 0)
})

test('empty canvas, dots and horizontal/vertical/diagonal lines are negative', () => {
  const model = new MLP(bundled.model)
  const paths = [[], [[[8, 8]]], [[[.1, .1], [15.9, 15.9]]], [[[.1, 15.9], [15.9, .1]]]]
  for (let position = .5; position < 16; position += .5) {
    paths.push([[[.1, position], [15.9, position]]], [[[position, .1], [position, 15.9]]])
  }
  for (const width of [.35, .67, 1, 1.4]) for (const path of paths) {
    const p = model.forward(rasterize(path, width)).at(-1)[0]
    assert.ok(p < .5, `line/dot/blank scored ${p}`)
  }
})

test('bundled weights meet held-out accuracy and line false-positive targets', () => {
  const model = new MLP(bundled.model), random = seededRandom(654321)
  const result = evaluate(model, trainingSet(2000, random))
  const lines = evaluate(model, Array.from({ length: 2000 }, () => makeExample('line', random)))
  assert.ok(result.accuracy >= .97, `accuracy ${result.accuracy}`)
  assert.ok(result.recall >= .97, `heart recall ${result.recall}`)
  assert.ok(lines.falsePositiveRate <= .005, `line FPR ${lines.falsePositiveRate}`)
})

test('retraining returns a checkpoint no worse than the initial validation loss', () => {
  const validation = trainingSet(800, seededRandom(92021))
  const before = evaluate(new MLP(bundled.model), validation)
  const iterator = trainModel(bundled.model, 34567, 2)
  let next = iterator.next(), progress = 0
  while (!next.done) { progress++; assert.ok(Number.isFinite(next.value.metrics.loss)); next = iterator.next() }
  assert.equal(progress, 2)
  const after = evaluate(new MLP(next.value.model), validation)
  assert.ok(after.loss <= before.loss + 1e-8)
})
