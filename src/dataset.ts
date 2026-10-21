export type Point = [number, number]
export type Example = { pixels: Float32Array; label: number; kind: string }
export const NEGATIVES = ['line', 'line', 'line', 'blank', 'dot', 'circle', 'square', 'triangle', 'star', 'cross', 'scribble', 'arc', 'partial-heart'] as const
type Kind = typeof NEGATIVES[number] | 'heart'

// Area-average rasterization at 64 × 64 matches the canvas preprocessing.
// Coordinates and stroke widths are expressed in 16 × 16 input-pixel units.
export function rasterize(paths: Point[][], width: number, ink = .88): Float32Array {
  const high = new Float32Array(64 * 64), result = new Float32Array(256), radius = width * 2
  for (const path of paths) for (let segment = 0; segment < path.length; segment++) {
    const a = path[Math.max(0, segment - 1)], b = path[segment]
    const ax = a[0] * 4, ay = a[1] * 4, bx = b[0] * 4, by = b[1] * 4, dx = bx - ax, dy = by - ay, length2 = dx * dx + dy * dy
    for (let y = Math.max(0, Math.floor(Math.min(ay, by) - radius - 1)); y < Math.min(64, Math.ceil(Math.max(ay, by) + radius + 1)); y++) {
      for (let x = Math.max(0, Math.floor(Math.min(ax, bx) - radius - 1)); x < Math.min(64, Math.ceil(Math.max(ax, bx) + radius + 1)); x++) {
        const t = length2 ? Math.max(0, Math.min(1, ((x + .5 - ax) * dx + (y + .5 - ay) * dy) / length2)) : 0
        const coverage = Math.max(0, Math.min(1, radius + .5 - Math.hypot(x + .5 - ax - t * dx, y + .5 - ay - t * dy)))
        high[y * 64 + x] = Math.max(high[y * 64 + x], coverage * ink)
      }
    }
  }
  for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) result[(y >> 2) * 16 + (x >> 2)] += high[y * 64 + x] / 16
  return result
}

export function makeExample(kind: Kind, random = Math.random): Example {
  const range = (a: number, b: number) => a + random() * (b - a)
  const angle = range(-.8, .8), sx = range(.55, 1.12), sy = sx * range(.8, 1.18), cx = range(6.3, 9.7), cy = range(6.3, 9.7)
  const width = range(.4, 1.3), ink = range(.75, 1)
  const transform = (points: Point[]) => points.map(([x, y]): Point => [cx + x * sx * Math.cos(angle) - y * sy * Math.sin(angle), cy + x * sx * Math.sin(angle) + y * sy * Math.cos(angle)])
  const close = (p: Point[]) => [...p, p[0]]
  let paths: Point[][] = []
  if (kind === 'heart' || kind === 'partial-heart') {
    const asymmetry = range(-.18, .18), notch = range(.7, 1.35)
    let points: Point[]
    if (random() < .5) {
      points = Array.from({ length: 80 }, (_, i) => {
        const t = i / 80 * Math.PI * 2, x = 5.4 * Math.sin(t) ** 3
        return [x, -(4.3 * Math.cos(t) - 1.7 * notch * Math.cos(2 * t) - .65 * Math.cos(3 * t) - .3 * Math.cos(4 * t)) + x * asymmetry]
      })
    } else {
      // A second heart family uses cubic curves, like a hand-drawn outline.
      const curves: Point[][] = [
        [[0, -2], [-2, -6.7 * notch], [-7, -4], [-4.7, -.2]],
        [[-4.7, -.2], [-3.7, 1.5], [-1, 3.7], [0, 5.3]],
        [[0, 5.3], [1.5, 3.5], [4.4, 1.4], [5, -.7]],
        [[5, -.7], [6.6, -4.5], [2.4, -6.3 * notch], [0, -2]],
      ]
      points = curves.flatMap(p => Array.from({ length: 20 }, (_, i): Point => {
        const t = i / 20, u = 1 - t
        const x = u ** 3 * p[0][0] + 3 * u * u * t * p[1][0] + 3 * u * t * t * p[2][0] + t ** 3 * p[3][0]
        const y = u ** 3 * p[0][1] + 3 * u * u * t * p[1][1] + 3 * u * t * t * p[2][1] + t ** 3 * p[3][1]
        return [x, y + x * asymmetry]
      }))
    }
    const wobble = range(0, .2), phase = range(0, 6.28)
    points = points.map(([x, y], i) => [x + wobble * Math.sin(i / 7 + phase), y + wobble * Math.cos(i / 9 + phase)])
    if (kind === 'partial-heart') {
      const start = Math.floor(random() * points.length), count = Math.floor(range(.15, .6) * points.length)
      paths = [transform(Array.from({ length: count }, (_, i) => points[(start + i) % points.length]))]
    } else paths = [transform(close(points))]
  } else if (kind === 'line') {
    // Arbitrary endpoints include short, off-center and edge-to-edge strokes.
    paths = [[[range(.1, 15.9), range(.1, 15.9)], [range(.1, 15.9), range(.1, 15.9)]]]
  } else if (kind === 'dot') paths = [[[range(0, 16), range(0, 16)]]]
  else if (kind === 'circle' || kind === 'arc') {
    const fraction = kind === 'arc' ? range(.2, .8) : 1, phase = range(0, 6.28)
    paths = [transform(Array.from({ length: 61 }, (_, i): Point => [5 * Math.cos(phase + i / 60 * Math.PI * 2 * fraction), 5 * Math.sin(phase + i / 60 * Math.PI * 2 * fraction)]))]
  } else if (kind === 'square') paths = [transform(close([[-5, -5], [5, -5], [5, 5], [-5, 5]]))]
  else if (kind === 'triangle') paths = [transform(close([[0, -5], [5, 5], [-5, 5]]))]
  else if (kind === 'star') paths = [transform(close(Array.from({ length: 10 }, (_, i): Point => {
    const t = i * Math.PI / 5 - Math.PI / 2, r = i % 2 ? 2.3 : 5.5
    return [Math.cos(t) * r, Math.sin(t) * r]
  })))]
  else if (kind === 'cross') paths = [transform([[-5, -5], [5, 5]]), transform([[-5, 5], [5, -5]])]
  else if (kind === 'scribble') paths = [Array.from({ length: Math.floor(range(2, 7)) }, (): Point => [range(1, 15), range(1, 15)])]
  return { pixels: rasterize(paths, width, ink), label: Number(kind === 'heart'), kind }
}

export function trainingSet(size: number, random = Math.random): Example[] {
  return Array.from({ length: size }, (_, i) => makeExample(i % 2 === 0 ? 'heart' : NEGATIVES[Math.floor(random() * NEGATIVES.length)], random))
}
