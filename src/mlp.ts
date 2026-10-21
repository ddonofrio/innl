export const ARCHITECTURE = [256, 64, 32, 16, 8, 1] as const
export const MODEL_VERSION = 3
export type Activations = Float32Array[]
export type SerializedModel = { version: number; sizes: number[]; weights: number[][]; biases: number[][] }
export type Layer = { weights: Float32Array; biases: Float32Array }
const NO_DISABLED_LAYERS = new Set<number>()
const NO_OVERRIDES = new Map<string, number>()

export function seededRandom(seed: number) {
  return () => {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed)
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t
    return ((t ^ t >>> 14) >>> 0) / 4294967296
  }
}

export function validModel(value: unknown): value is SerializedModel {
  if (!value || typeof value !== 'object') return false
  const v = value as SerializedModel
  return v.version === MODEL_VERSION && JSON.stringify(v.sizes) === JSON.stringify(ARCHITECTURE)
    && Array.isArray(v.weights) && v.weights.length === 5 && Array.isArray(v.biases) && v.biases.length === 5
    && ARCHITECTURE.slice(1).every((size, l) =>
      Array.isArray(v.weights[l]) && v.weights[l].length === size * ARCHITECTURE[l] && v.weights[l].every(Number.isFinite)
      && Array.isArray(v.biases[l]) && v.biases[l].length === size && v.biases[l].every(Number.isFinite))
}

export class MLP {
  layers: Layer[]
  private moments: { m: Float32Array; v: Float32Array }[]
  private step = 0

  constructor(saved?: SerializedModel, random = Math.random) {
    if (saved && !validModel(saved)) throw new Error('Incompatible or corrupt model')
    this.layers = ARCHITECTURE.slice(1).map((size, l) => ({
      weights: saved ? new Float32Array(saved.weights[l]) : Float32Array.from({ length: size * ARCHITECTURE[l] }, () =>
        Math.sqrt(-2 * Math.log(Math.max(1e-8, random()))) * Math.cos(2 * Math.PI * random()) * Math.sqrt(2 / ARCHITECTURE[l])),
      biases: saved ? new Float32Array(saved.biases[l]) : new Float32Array(size),
    }))
    this.moments = this.layers.flatMap(l => [l.weights, l.biases]).map(p => ({ m: new Float32Array(p.length), v: new Float32Array(p.length) }))
  }

  get parameterCount() { return this.layers.reduce((n, l) => n + l.weights.length + l.biases.length, 0) }

  forward(input: Float32Array, disabledLayers: ReadonlySet<number> = NO_DISABLED_LAYERS, overrides: ReadonlyMap<string, number> = NO_OVERRIDES): Activations {
    const inputActivation = overrides.size ? new Float32Array(input) : input
    if (overrides.size) {
      for (let i = 0; i < inputActivation.length; i++) {
        const value = overrides.get(`0:${i}`)
        if (value !== undefined) inputActivation[i] = Math.max(0, Math.min(1, value))
      }
    }
    const activations = [inputActivation]
    this.layers.forEach((layer, l) => {
      const source = activations[l], output = new Float32Array(layer.biases.length)
      if (!disabledLayers.has(l + 1)) {
        for (let j = 0; j < output.length; j++) {
          let sum = layer.biases[j]
          for (let i = 0; i < source.length; i++) sum += source[i] * layer.weights[j * source.length + i]
          output[j] = l === this.layers.length - 1 ? 1 / (1 + Math.exp(-sum)) : Math.max(0, sum)
        }
        for (let j = 0; j < output.length; j++) {
          const value = overrides.get(`${l + 1}:${j}`)
          if (value !== undefined) output[j] = Math.max(0, l === this.layers.length - 1 ? Math.min(1, value) : value)
        }
      }
      activations.push(output)
    })
    return activations
  }

  // BCE + sigmoid has derivative p - y. Hidden deltas use the ReLU derivative.
  gradients(samples: Float32Array[], labels: number[]) {
    const gradients = this.layers.map(l => ({ weights: new Float32Array(l.weights.length), biases: new Float32Array(l.biases.length) }))
    let loss = 0, correct = 0
    for (let s = 0; s < samples.length; s++) {
      const a = this.forward(samples[s]), p = a.at(-1)![0], y = labels[s]
      loss -= y * Math.log(Math.max(1e-7, p)) + (1 - y) * Math.log(Math.max(1e-7, 1 - p))
      correct += Number((p >= .5) === (y === 1))
      let delta: Float32Array = new Float32Array([p - y])
      for (let l = this.layers.length - 1; l >= 0; l--) {
        const source = a[l], g = gradients[l], weights = this.layers[l].weights, previous = new Float32Array(source.length)
        for (let j = 0; j < delta.length; j++) {
          g.biases[j] += delta[j] / samples.length
          for (let i = 0; i < source.length; i++) {
            g.weights[j * source.length + i] += delta[j] * source[i] / samples.length
            previous[i] += weights[j * source.length + i] * delta[j]
          }
        }
        for (let i = 0; i < previous.length; i++) if (source[i] <= 0) previous[i] = 0
        delta = previous
      }
    }
    return { gradients, loss: loss / samples.length, correct }
  }

  // Adam is adaptive mini-batch gradient descent; L2 limits very large weights.
  trainBatch(samples: Float32Array[], labels: number[], learningRate = .001) {
    const result = this.gradients(samples, labels)
    const params = this.layers.flatMap(l => [l.weights, l.biases])
    const gradients = result.gradients.flatMap(l => [l.weights, l.biases])
    this.step++
    const c1 = 1 - .9 ** this.step, c2 = 1 - .999 ** this.step
    params.forEach((p, k) => {
      const { m, v } = this.moments[k]
      for (let i = 0; i < p.length; i++) {
        const g = Math.max(-5, Math.min(5, gradients[k][i] + (k % 2 === 0 ? .0001 * p[i] : 0)))
        m[i] = .9 * m[i] + .1 * g; v[i] = .999 * v[i] + .001 * g * g
        p[i] -= learningRate * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8)
      }
    })
    return { loss: result.loss, correct: result.correct }
  }

  serialize(): SerializedModel {
    return { version: MODEL_VERSION, sizes: [...ARCHITECTURE], weights: this.layers.map(l => [...l.weights]), biases: this.layers.map(l => [...l.biases]) }
  }
}
