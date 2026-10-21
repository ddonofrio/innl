import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ARCHITECTURE, MLP, seededRandom, validModel } from './mlp'
import NetworkView from './NetworkView'
import type { Evaluation, Metrics, TrainingMessage, TrainingResult } from './training'

const STORAGE_KEY = 'heart-mlp-v3'

function validResult(value: unknown): value is TrainingResult {
  if (!value || typeof value !== 'object') return false
  const result = value as TrainingResult
  return validModel(result.model) && Number.isInteger(result.bestEpoch) && Array.isArray(result.history)
    && result.history.every(m => ['epoch', 'loss', 'accuracy', 'falsePositiveRate', 'recall', 'trainLoss', 'trainAccuracy'].every(k => Number.isFinite(m[k as keyof Metrics])))
    && !!result.validation && ['loss', 'accuracy', 'falsePositiveRate', 'recall'].every(k => Number.isFinite(result.validation[k as keyof Evaluation]))
}

function DrawingPad({ onInput, input }: { onInput: (x: Float32Array) => void; input: Float32Array }) {
  const ref = useRef<HTMLCanvasElement>(null), drawing = useRef(false), last = useRef<{ x: number; y: number } | null>(null)
  const paint = useCallback((clear = false) => {
    const canvas = ref.current; if (!canvas) return
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!
    if (clear) { ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, canvas.width, canvas.height) }
    // Exact area averaging: each input pixel averages a 21 × 21 block.
    // White is zero and dark ink is positive, as in the training rasterizer.
    const data = ctx.getImageData(0, 0, 336, 336).data, pixels = new Float32Array(256)
    for (let y = 0; y < 336; y++) for (let x = 0; x < 336; x++) {
      const at = (y * 336 + x) * 4
      pixels[Math.floor(y / 21) * 16 + Math.floor(x / 21)] += (1 - (data[at] + data[at + 1] + data[at + 2]) / 765) / 441
    }
    onInput(pixels)
  }, [onInput])
  useEffect(() => { paint(true) }, [paint])
  const coords = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const r = event.currentTarget.getBoundingClientRect()
    return { x: (event.clientX - r.left - 1) * 336 / (r.width - 2), y: (event.clientY - r.top - 1) * 336 / (r.height - 2) }
  }
  const start = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (event.button !== 0) return
    event.currentTarget.setPointerCapture(event.pointerId); drawing.current = true; last.current = coords(event)
    const ctx = event.currentTarget.getContext('2d')!
    ctx.fillStyle = '#1d2531'; ctx.beginPath(); ctx.arc(last.current.x, last.current.y, 7, 0, Math.PI * 2); ctx.fill(); paint()
  }
  const move = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawing.current) return
    const p = coords(event), ctx = ref.current!.getContext('2d')!
    ctx.strokeStyle = '#1d2531'; ctx.lineWidth = 14; ctx.lineCap = 'round'; ctx.lineJoin = 'round'
    ctx.beginPath(); ctx.moveTo(last.current!.x, last.current!.y); ctx.lineTo(p.x, p.y); ctx.stroke(); last.current = p; paint()
  }
  const end = () => { drawing.current = false; last.current = null }
  return <section className="panel draw-panel">
    <div className="panel-heading"><div><p className="eyebrow">Live input</p><h2>Draw a shape</h2></div><button className="ghost" onClick={() => { end(); paint(true) }}>Clear canvas</button></div>
    <div className="drawing-preview">
      <div><div className="matrix-title"><span>Drawing canvas</span></div><canvas aria-label="Draw a heart or another shape" ref={ref} width="336" height="336" className="drawing-canvas" onPointerDown={start} onPointerMove={move} onPointerUp={end} onPointerCancel={end} onLostPointerCapture={end} /></div>
      <div><div className="matrix-title"><span>Processed input</span><span>16 × 16</span></div><div className="input-grid" aria-label="16 by 16 input pixels">{Array.from(input, (value, i) => <i key={i} style={{ background: `rgb(${255 * (1 - value)}, ${255 * (1 - value)}, ${255 * (1 - value)})` }} />)}</div></div>
    </div>
    <p className="hint">Draw a complete outline. The input grid and model update immediately, including during training.</p>
  </section>
}

function TrainingChart({ history, bestEpoch, training }: { history: Metrics[]; bestEpoch: number; training: boolean }) {
  const latest = history.at(-1), maxLoss = Math.max(.1, ...history.map(m => Math.max(m.loss, m.trainLoss))) * 1.1
  const points = (key: 'loss' | 'trainLoss' | 'accuracy', max: number) => history.map((m, i) => `${i / Math.max(1, history.length - 1) * 100},${56 - m[key] / max * 50}`).join(' ')
  return <section className="panel training-panel">
    <div className="panel-heading"><div><p className="eyebrow">Supervised learning · unseen validation examples</p><h2>Training telemetry</h2></div>
      {latest && <span className="metrics">Epoch {latest.epoch}{training ? ' / 60' : bestEpoch ? ` · restored best epoch ${bestEpoch}` : ' · retained starting checkpoint'}</span>}
    </div>
    {latest ? <>
      <div className="training-stats"><span>Validation accuracy <strong>{(latest.accuracy * 100).toFixed(2)}%</strong></span><span>Validation loss <strong>{latest.loss.toFixed(4)}</strong></span><span>False positives <strong>{(latest.falsePositiveRate * 100).toFixed(2)}%</strong></span></div>
      <div className="chart-pair"><div><div className="chart-axis">Cross-entropy loss · 0–{maxLoss.toFixed(2)}</div><svg viewBox="0 0 100 60" className="chart" preserveAspectRatio="none" role="img" aria-label="Training and validation loss by epoch"><path d="M0 56H100" /><polyline points={points('trainLoss', maxLoss)} className="train-loss-line" /><polyline points={points('loss', maxLoss)} className="loss-line" /></svg><div className="chart-legend"><span><i className="train-dot" /> Training</span><span><i className="loss-dot" /> Validation</span></div></div>
        <div><div className="chart-axis">Validation accuracy · 0–100%</div><svg viewBox="0 0 100 60" className="chart" preserveAspectRatio="none" role="img" aria-label="Validation accuracy by epoch"><path d="M0 56H100" /><polyline points={points('accuracy', 1)} className="accuracy-line" /></svg><div className="chart-legend"><span><i className="accuracy-dot" /> Unseen validation set</span></div></div></div>
      <p className="hint">2,048 fresh examples per epoch · 800 fixed validation examples · Adam + L2 regularization · early stopping. Metrics above describe the last epoch; the saved model uses the best validation loss.</p>
    </> : <p className="hint">Training generates new hearts, lines, dots, blank images, incomplete outlines and other shapes. Validation examples are never used for gradient updates.</p>}
  </section>
}

export default function App() {
  const [model, setModel] = useState(() => new MLP(undefined, seededRandom(8427)))
  const [input, setInput] = useState<Float32Array>(() => new Float32Array(256))
  const [history, setHistory] = useState<Metrics[]>([]), [bestEpoch, setBestEpoch] = useState(0)
  const [ready, setReady] = useState(false), [training, setTraining] = useState(false), [trained, setTrained] = useState(false)
  const [hasSavedModel, setHasSavedModel] = useState(false), [disabledLayers, setDisabledLayers] = useState<Set<number>>(() => new Set())
  const [neuronOverrides, setNeuronOverrides] = useState<Map<string, number>>(() => new Map())
  const [status, setStatus] = useState('Loading the improved pretrained model…')
  const worker = useRef<Worker | null>(null)
  const stable = useRef<{ model: MLP; history: Metrics[]; bestEpoch: number } | null>(null)
  const bundledModel = useRef<TrainingResult | null>(null)
  const activations = useMemo(() => model.forward(input, disabledLayers, neuronOverrides), [model, input, disabledLayers, neuronOverrides])
  const infer = useCallback((pixels: Float32Array) => setInput(pixels), [])

  useEffect(() => {
    const controller = new AbortController()
    const load = async () => {
      let storageAvailable = true
      try {
        let stored: unknown
        try { stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null') } catch { storageAvailable = false }
        const response = await fetch(`${import.meta.env.BASE_URL}model-v3.json`, { signal: controller.signal })
        if (!response.ok) throw new Error('Bundled model could not be loaded')
        const defaultResult: unknown = await response.json()
        if (!validResult(defaultResult)) throw new Error('The bundled model file is invalid')
        bundledModel.current = defaultResult
        const savedResult = validResult(stored) ? stored : null
        const saved = savedResult !== null, result: TrainingResult = savedResult ?? defaultResult
        if (controller.signal.aborted) return
        setModel(new MLP(result.model)); setHistory(result.history); setBestEpoch(result.bestEpoch); setTrained(true)
        setHasSavedModel(saved)
        setStatus(`${saved ? 'Saved retrained model' : 'Improved pretrained model'} loaded · ${(result.validation.accuracy * 100).toFixed(2)}% validation accuracy.${storageAvailable ? '' : ' Browser storage is unavailable.'}`)
      } catch (error) {
        if (controller.signal.aborted) return
        setStatus(`${error instanceof Error ? error.message : 'Loading failed'}. Click Retrain model to train locally.`)
      }
      setReady(true)
    }
    void load()
    return () => { controller.abort(); worker.current?.terminate() }
  }, [])

  const stop = () => {
    worker.current?.terminate(); worker.current = null
    if (stable.current) { setModel(stable.current.model); setHistory(stable.current.history); setBestEpoch(stable.current.bestEpoch) }
    setTraining(false); setStatus('Training stopped. The previous model has been restored.')
  }
  const train = () => {
    if (worker.current || !ready) return
    stable.current = { model, history, bestEpoch }
    setDisabledLayers(new Set()); setNeuronOverrides(new Map()); setTraining(true); setHistory([]); setStatus('Training in a background worker. You can keep drawing while the network animates each epoch.')
    const fail = (message: string) => {
      worker.current?.terminate(); worker.current = null
      if (stable.current) { setModel(stable.current.model); setHistory(stable.current.history); setBestEpoch(stable.current.bestEpoch) }
      setTraining(false); setStatus(`Training failed: ${message}. The previous model has been restored.`)
    }
    try {
      const active = new Worker(new URL('./training.worker.ts', import.meta.url), { type: 'module' }); worker.current = active
      active.onerror = event => fail(event.message)
      active.onmessage = (event: MessageEvent<TrainingMessage>) => {
        const message = event.data
        if (message.type === 'error') { fail(message.message); return }
        if (message.type === 'progress') {
          setModel(new MLP(message.model)); setHistory(h => [...h, message.metrics]); return
        }
        const result = message.result
        setModel(new MLP(result.model)); setHistory(result.history); setBestEpoch(result.bestEpoch); setTrained(true); setTraining(false)
        let persisted = true
        try { localStorage.setItem(STORAGE_KEY, JSON.stringify(result)) } catch { persisted = false }
        setHasSavedModel(persisted)
        setStatus(`Best checkpoint restored · ${(result.validation.accuracy * 100).toFixed(2)}% validation accuracy. ${persisted ? 'Weights and training history saved in this browser.' : 'Storage unavailable: this model will not survive a reload.'}`)
        active.terminate(); worker.current = null
      }
      active.postMessage({ model: model.serialize(), seed: Date.now() >>> 0 })
    } catch (error) { fail(error instanceof Error ? error.message : 'Worker unavailable') }
  }

  const resetToDefault = () => {
    const result = bundledModel.current
    if (!result || training) return
    try { localStorage.removeItem(STORAGE_KEY); localStorage.removeItem('heart-mlp-v1') } catch { /* The model can still reset for this session. */ }
    setModel(new MLP(result.model)); setHistory(result.history); setBestEpoch(result.bestEpoch); setDisabledLayers(new Set()); setNeuronOverrides(new Map()); setHasSavedModel(false); setTrained(true)
    setStatus(`Default pretrained model restored · ${(result.validation.accuracy * 100).toFixed(2)}% validation accuracy. Saved retraining data was removed.`)
  }

  const toggleLayer = (layer: number) => setDisabledLayers(current => {
    const next = new Set(current); if (next.has(layer)) next.delete(layer); else next.add(layer); return next
  })
  const setNeuronOverride = (layer: number, index: number, value?: number) => setNeuronOverrides(current => {
    const next = new Map(current), key = `${layer}:${index}`
    if (value === undefined) next.delete(key); else next.set(key, value)
    return next
  })

  const probability = activations.at(-1)![0]
  return <main>
    <header><div><p className="eyebrow">Interactive neural network laboratory</p><h1>Does this look like a <i>heart</i>?</h1><p className="intro">Draw a shape and follow its signal through four hidden layers. Every glow and connection comes from the real neural network.</p></div></header>
    <p className="model-status" role="status">{status}</p>
    <NetworkView model={model} activations={activations} disabledLayers={disabledLayers} onToggleLayer={toggleLayer} training={training} epoch={history.at(-1)?.epoch ?? 0} neuronOverrides={neuronOverrides} onOverrideNeuron={setNeuronOverride} onClearNeuronOverrides={() => setNeuronOverrides(new Map())} />
    <div className="top-grid"><DrawingPad onInput={infer} input={input} /><section className="panel prediction-panel"><p className="eyebrow">Classifier output{training ? ' · training preview' : ''}</p><div className="probability"><strong><span>{ready && (trained || training) ? (probability * 100).toFixed(1) : '—'}</span><small>%</small></strong><span>heart probability</span></div><div className="meter"><i style={{ width: `${ready && (trained || training) ? probability * 100 : 0}%` }} /></div><p className={probability >= .5 ? 'verdict yes' : 'verdict'}>{!ready ? 'Loading model…' : !trained && !training ? 'Model needs training' : probability >= .5 ? 'Likely a heart' : 'Not recognised as a heart'}</p><div className="model-actions"><button className={`ghost${training ? ' training-active' : ''}`} onClick={training ? stop : train} disabled={!ready}>{training ? 'Stop training' : 'Retrain model'}</button><button className="ghost reset-button" onClick={resetToDefault} disabled={!ready || training || !hasSavedModel}>Reset to default</button></div><p className="model-note">{ARCHITECTURE.join(' → ')}<br />{model.parameterCount.toLocaleString('en-US')} trainable parameters</p><p className="hint">{neuronOverrides.size ? `${neuronOverrides.size} manual neuron value${neuronOverrides.size > 1 ? 's are' : ' is'} affecting this prediction.` : disabledLayers.size ? `${disabledLayers.size} hidden layer${disabledLayers.size > 1 ? 's are' : ' is'} deactivated for this prediction.` : 'Trained on synthetic outlines. Unusual hand-drawn shapes can still be misclassified.'}</p></section></div>
    <TrainingChart history={history} bestEpoch={bestEpoch} training={training} />
    <footer>Real MLP · ReLU hidden layers · sigmoid output · binary cross-entropy · backpropagation with Adam. Retraining continues from the current weights and saves the best validation checkpoint in this browser.</footer>
  </main>
}
