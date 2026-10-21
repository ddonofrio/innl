# Heart MLP Lab

An educational React application that trains a real multilayer perceptron to distinguish hand-drawn heart outlines from other shapes. Inference, retraining, backpropagation and the animated network visualization all run locally.

## Run

```bash
npm install
npm run dev
```

Useful checks:

```bash
npm test
npm run build
npm run train
```

The network is `256 → 64 → 32 → 16 → 8 → 1` with 19,201 trainable parameters. It uses ReLU hidden layers, a sigmoid output, binary cross-entropy and Adam. The included pretrained weights are stored in `public/model-v3.json`; browser retraining saves the best validation checkpoint to `localStorage`.

The visualizer displays actual weights and activations. Forward pulses represent large `activation × weight` contributions, purple training pulses represent the weights that changed most between epochs, and the eye controls perform real ablation by setting a hidden layer's activations to zero. Click a neuron to override its activation and propagate that intervention forward. Pause or scrub the Flow timeline to inspect the animation at any point.

See `public/model-report.json` for the bundled model's held-out synthetic evaluation. These measurements do not guarantee performance on every hand-drawn shape.
