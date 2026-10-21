import { useEffect, useRef, useState } from 'react'
import { ARCHITECTURE, MLP, type Activations } from './mlp'

type Position = { x: number; y: number }
type Signal = { layer: number; from: number; to: number; contribution: number }
const layerName = (l: number) => l === 0 ? 'Input' : l === 5 ? 'Output' : `Hidden ${l}`
const layerDetail = (l: number) => l === 0 ? '256 pixels' : l === 5 ? '1 sigmoid' : `${ARCHITECTURE[l]} ReLU`

function EyeIcon({ open }: { open: boolean }) {
  return open
    ? <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z" /><circle cx="12" cy="12" r="2.7" /></svg>
    : <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4.2 7.4C3.1 8.7 2.5 10 2.5 10s3.5 6 9.5 6c1.4 0 2.7-.3 3.8-.8M7.2 5.1A10 10 0 0 1 12 4c6 0 9.5 6 9.5 6a14 14 0 0 1-2.3 3.1M9.4 9.4a3.7 3.7 0 0 0 5.2 5.2M3 3l18 18" /></svg>
}

export default function NetworkView({ model, activations, disabledLayers, onToggleLayer, training, epoch, neuronOverrides, onOverrideNeuron, onClearNeuronOverrides }: { model: MLP; activations: Activations; disabledLayers: ReadonlySet<number>; onToggleLayer: (layer: number) => void; training: boolean; epoch: number; neuronOverrides: ReadonlyMap<string, number>; onOverrideNeuron: (layer: number, index: number, value?: number) => void; onClearNeuronOverrides: () => void }) {
  const canvas = useRef<HTMLCanvasElement>(null), wrap = useRef<HTMLDivElement>(null)
  const state = useRef({ activations, previous: activations, changedAt: performance.now(), model })
  const positions = useRef<Position[][]>([]), signals = useRef<Signal[]>([]), trainingSignals = useRef<Signal[]>([])
  const playhead = useRef(0), timeline = useRef<HTMLInputElement>(null), timelineLabel = useRef<HTMLOutputElement>(null)
  const [playing, setPlaying] = useState(() => !window.matchMedia('(prefers-reduced-motion: reduce)').matches)
  const [speed, setSpeed] = useState(1), [selected, setSelected] = useState<{ layer: number; index: number } | null>(null)
  useEffect(() => {
    const previousModel = state.current.model
    state.current = { activations, previous: state.current.activations, changedAt: performance.now(), model }
    // Animate the two largest actual signed contributions into each destination.
    // All weights remain visible in the background, not only this animated subset.
    signals.current = model.layers.flatMap((layer, l) => {
      const selected: Signal[] = []
      for (let j = 0; j < layer.biases.length; j++) {
        const contributions = Array.from(activations[l], (a, i) => ({ layer: l, from: i, to: j, contribution: a * layer.weights[j * activations[l].length + i] }))
        contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
        selected.push(...contributions.slice(0, 2).filter(s => Math.abs(s.contribution) > .0001))
      }
      return selected
    })
    trainingSignals.current = training ? model.layers.flatMap((layer, l) => {
      const old = previousModel.layers[l]?.weights
      if (!old || old.length !== layer.weights.length) return []
      return Array.from(layer.weights, (weight, index) => ({ layer: l, from: index % activations[l].length, to: Math.floor(index / activations[l].length), contribution: weight - old[index] }))
        .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, 24)
    }) : []
  }, [model, activations, training])

  useEffect(() => {
    const c = canvas.current!, element = wrap.current!, g = c.getContext('2d')!
    const background = document.createElement('canvas'), bg = background.getContext('2d')!
    let w = 0, h = 0, dpr = 1, frame = 0, last = 0, previousFrame = performance.now()
    const rebuild = () => {
      w = element.clientWidth; h = element.clientHeight; dpr = Math.min(devicePixelRatio || 1, 2)
      c.width = background.width = Math.round(w * dpr); c.height = background.height = Math.round(h * dpr)
      g.setTransform(dpr, 0, 0, dpr, 0, 0); bg.setTransform(dpr, 0, 0, dpr, 0, 0)
      positions.current = ARCHITECTURE.map((size, l) => Array.from({ length: size }, (_, i) => {
        const x = 80 + l * (w - 160) / 5
        if (l === 0) return { x: x + (i % 16 - 7.5) * 5, y: h / 2 + (Math.floor(i / 16) - 7.5) * 5 }
        const columns = size === 64 ? 2 : 1, rows = size / columns
        return { x: x + (i % columns - (columns - 1) / 2) * 13, y: rows === 1 ? h / 2 : 64 + Math.floor(i / columns) * (h - 112) / (rows - 1) }
      }))
      bg.fillStyle = '#0d1728'; bg.fillRect(0, 0, w, h)
      // Render every real weight once; only signals and neurons redraw per frame.
      model.layers.forEach((layer, l) => {
        const from = positions.current[l], to = positions.current[l + 1]
        for (let j = 0; j < to.length; j++) for (let i = 0; i < from.length; i++) {
          const weight = layer.weights[j * from.length + i], strength = Math.min(1, Math.abs(weight))
          bg.strokeStyle = weight >= 0 ? `rgba(255,166,95,${.009 + strength * .075})` : `rgba(83,178,255,${.009 + strength * .075})`
          bg.lineWidth = .25 + strength * .7; bg.beginPath(); bg.moveTo(from[i].x, from[i].y); bg.lineTo(to[j].x, to[j].y); bg.stroke()
        }
      })
    }
    const render = (now: number) => {
      frame = requestAnimationFrame(render)
      if (now - last < 32 || document.hidden) return
      const elapsed = Math.min(100, now - previousFrame); previousFrame = now; last = now
      g.clearRect(0, 0, w, h); g.drawImage(background, 0, 0, w, h)
      if (playing) playhead.current = (playhead.current + elapsed * speed / (650 * 5)) % 1
      const phase = Math.min(4.9999, playhead.current * 5)
      const stage = Math.floor(phase), progress = phase - stage
      if (timeline.current) timeline.current.value = String(Math.round(playhead.current * 1000))
      if (timelineLabel.current) timelineLabel.current.textContent = `${layerName(stage)} → ${layerName(stage + 1)}`
      for (const signal of signals.current) {
        if (signal.layer !== stage) continue
        const a = positions.current[stage][signal.from], b = positions.current[stage + 1][signal.to]
        const strength = Math.min(1, Math.abs(signal.contribution) / 1.5), color = signal.contribution >= 0 ? '#ffb871' : '#65cfff'
        const x = a.x + (b.x - a.x) * progress, y = a.y + (b.y - a.y) * progress
        g.globalAlpha = .2 + strength * .8; g.strokeStyle = color; g.lineWidth = .7 + strength
        g.beginPath(); g.moveTo(a.x + (b.x - a.x) * Math.max(0, progress - .16), a.y + (b.y - a.y) * Math.max(0, progress - .16)); g.lineTo(x, y); g.stroke()
        g.shadowColor = color; g.shadowBlur = 7 + strength * 9; g.fillStyle = color
        g.beginPath(); g.arc(x, y, 1 + strength * 2, 0, Math.PI * 2); g.fill()
      }
      if (training) {
        const backwardStage = 4 - stage
        for (const signal of trainingSignals.current) {
          if (signal.layer !== backwardStage) continue
          const from = positions.current[signal.layer][signal.from], to = positions.current[signal.layer + 1][signal.to]
          const strength = Math.min(1, Math.abs(signal.contribution) * 180), color = '#d7a7ff'
          const x = to.x + (from.x - to.x) * progress, y = to.y + (from.y - to.y) * progress
          g.globalAlpha = .35 + strength * .65; g.strokeStyle = color; g.lineWidth = 1 + strength * 1.8
          g.beginPath(); g.moveTo(to.x + (from.x - to.x) * Math.max(0, progress - .2), to.y + (from.y - to.y) * Math.max(0, progress - .2)); g.lineTo(x, y); g.stroke()
          g.shadowColor = color; g.shadowBlur = 10 + strength * 14; g.fillStyle = color
          g.beginPath(); g.arc(x, y, 1.4 + strength * 2.4, 0, Math.PI * 2); g.fill()
        }
      }
      g.shadowBlur = 0; g.globalAlpha = 1
      const current = state.current, fade = Math.max(0, 1 - (now - current.changedAt) / 650)
      positions.current.forEach((points, l) => points.forEach((p, i) => {
        const value = current.activations[l][i], level = l === 0 || l === 5 ? value : value / (1 + value)
        const difference = Math.min(1, Math.abs(value - current.previous[l][i])) * fade
        const radius = l === 0 ? 1.8 : l === 5 ? 13 : l === 1 ? 3 : 4.5
        const wave = l === stage && l > 0 ? 1 - progress : 0
        g.shadowColor = '#ffbe71'; g.shadowBlur = level * (8 + wave * 13) + difference * 16
        g.fillStyle = `rgb(${Math.round(40 + level * 215)},${Math.round(53 + level * 147)},${Math.round(73 + level * 42)})`
        g.beginPath(); g.arc(p.x, p.y, radius + difference * 2, 0, Math.PI * 2); g.fill()
        g.shadowBlur = 0; g.strokeStyle = `rgba(255,225,175,${.18 + level * .65})`; g.lineWidth = .75; g.stroke()
        if (difference > .03) {
          g.strokeStyle = `rgba(255,255,255,${difference})`; g.beginPath(); g.arc(p.x, p.y, radius + 4 + (1 - fade) * 8, 0, Math.PI * 2); g.stroke()
        }
        if (selected?.layer === l && selected.index === i) {
          g.strokeStyle = '#7ce8dc'; g.lineWidth = 2; g.shadowColor = '#7ce8dc'; g.shadowBlur = 10
          g.beginPath(); g.arc(p.x, p.y, radius + 6, 0, Math.PI * 2); g.stroke(); g.shadowBlur = 0
        }
      }))
    }
    rebuild(); const observer = new ResizeObserver(rebuild); observer.observe(element); frame = requestAnimationFrame(render)
    return () => { cancelAnimationFrame(frame); observer.disconnect() }
  }, [model, playing, speed, training, selected])

  const selectedKey = selected ? `${selected.layer}:${selected.index}` : ''
  const selectedValue = selected ? activations[selected.layer][selected.index] : 0
  const selectedOverridden = selected ? neuronOverrides.has(selectedKey) : false
  const selectedMax = !selected || selected.layer === 0 || selected.layer === 5 ? 1 : Math.max(2, Math.ceil(Math.max(selectedValue, ...activations[selected.layer]) * 1.35 * 10) / 10)

  return <section className="panel network-panel">
    <div className="panel-heading"><div><p className="eyebrow">Real model state · four hidden layers{training ? ` · training epoch ${epoch}` : ''}</p><h2>Activation flow</h2></div>
      <div className="flow-controls"><button className="ghost" onClick={() => setPlaying(p => !p)}>{playing ? 'Pause' : 'Play'}</button>
        <label className="timeline-control"><span>Flow</span><input ref={timeline} type="range" min="0" max="1000" defaultValue="0" aria-label="Animation position" onPointerDown={() => setPlaying(false)} onInput={event => { playhead.current = Number(event.currentTarget.value) / 1000 }} /><output ref={timelineLabel}>Input → Hidden 1</output></label>
        <label>Speed <select value={speed} onChange={e => setSpeed(Number(e.target.value))}><option value={.5}>0.5×</option><option value={1}>1×</option><option value={2}>2×</option></select></label></div>
    </div>
    <div className="network-scroll"><div className="network-wrap" ref={wrap}>
      <div className="network-layer-headings">{ARCHITECTURE.map((_, layer) => <div key={layer} className={`network-layer-heading layer-${layer}`}><span>{layerName(layer)}{layer > 0 && layer < 5 && <button className={`eye-toggle ${disabledLayers.has(layer) ? 'eye-off' : ''}`} disabled={training} aria-pressed={!disabledLayers.has(layer)} aria-label={`${disabledLayers.has(layer) ? 'Enable' : 'Disable'} Hidden ${layer}`} title={`${disabledLayers.has(layer) ? 'Enable' : 'Disable'} Hidden ${layer}`} onClick={() => onToggleLayer(layer)}><EyeIcon open={!disabledLayers.has(layer)} /></button>}</span><em>{layerDetail(layer)}</em></div>)}</div>
      <canvas ref={canvas} role="img" aria-label="Six network layers. Click a neuron to edit its activation. Orange connections have positive weights, blue connections have negative weights." onClick={event => {
        const box = event.currentTarget.getBoundingClientRect(), x = event.clientX - box.left, y = event.clientY - box.top
        let nearest: { layer: number; index: number } | null = null, distance = 15
        positions.current.forEach((points, layer) => points.forEach((p, index) => { const d = Math.hypot(p.x - x, p.y - y); if (d < distance) { distance = d; nearest = { layer, index } } }))
        if (nearest) setSelected(nearest)
      }} />
    </div></div>
    {selected && <div className="neuron-editor">
      <div><p className="eyebrow">Neuron intervention</p><strong>{layerName(selected.layer)} · neuron {selected.index + 1}</strong><span>{selectedOverridden ? 'Manual activation' : 'Calculated activation'}</span></div>
      <label><input type="range" min="0" max={selectedMax} step={selected.layer === 0 || selected.layer === 5 ? .01 : .02} value={selectedValue} disabled={training || disabledLayers.has(selected.layer)} onChange={event => onOverrideNeuron(selected.layer, selected.index, Number(event.currentTarget.value))} /><output>{selectedValue.toFixed(3)}</output></label>
      <div className="editor-actions"><button className="ghost" disabled={!selectedOverridden || training} onClick={() => onOverrideNeuron(selected.layer, selected.index)}>Restore this value</button><button className="ghost" disabled={!neuronOverrides.size || training} onClick={onClearNeuronOverrides}>Restore all values</button></div>
      <button className="editor-close" aria-label="Close neuron editor" onClick={() => setSelected(null)}>×</button>
    </div>}
  </section>
}
