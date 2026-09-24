# GPT-2 Shakespeare Example

Train a small character-level GPT-2 model on Tiny Shakespeare with Magma's
Metal backend. This package is experimental: see [../README.md](../README.md).

## Requirements

- macOS 15 or later on Apple Silicon. The model runs on the Metal backend
  only, and the program exits 1 if Metal or MetalHLO is unavailable.
- Magma built with its Metal backend: set `MAGMA_ENABLE_METAL=1`, and
  optionally `MAGMA_METALHLO_PATH=/path/to/MetalHLO` to use a local MetalHLO
  checkout.
- Python 3 and network access for the one-time data preparation.

## Quick Start

```bash
cd Benchmarks/GPT2Shakespeare

# 1. Prepare data. This downloads Tiny Shakespeare and writes data/train.bin,
#    data/val.bin and data/meta.json. data/ is git-ignored.
python3 prepare_data.py

# 2. Train. Run from this directory, because the data path is relative.
MAGMA_ENABLE_METAL=1 swift run -c release GPT2Shakespeare
```

The program always trains `GPT2Config.shakespeareTiny` with
`TrainConfig.shakespeare` (5000 steps). To try the smaller `test` model or the
50-step `quick` schedule, edit the two lines in `trainShakespeare()` in
`Sources/main.swift` that select them.

## What This Demonstrates

- **GPT-2 architecture**: a transformer decoder with
  - token and position embeddings
  - multi-head causal self-attention
  - a feed-forward MLP with GELU activation
  - pre-norm layer normalization
- **Training features from Magma**:
  - reverse-mode autodiff through the whole model (`valueWithGradient`)
  - `optim.Adam` with weight decay, stepped with gradients keyed
    by `Parameter`
  - gradient clipping (`optim.clipGradNorm`)
  - learning-rate warmup and cosine decay (`optim.WarmupCosineScheduler`)
- **Data pipeline**: memory-mapped token files (`MemoryMappedTokenDataset`)
  with character-level tokenization

## Model Configurations

| Config | Layers | Heads | Embed | Block | Params |
|--------|--------|-------|-------|-------|--------|
| `test` | 2 | 2 | 64 | 64 | ~0.1M |
| `shakespeareTiny` (default) | 6 | 6 | 384 | 256 | ~10M |

## Files

- `Sources/main.swift`: the GPT-2 model and training loop
- `prepare_data.py`: the data preparation script
- `data/`: downloaded and tokenized data, created by `prepare_data.py` and
  not committed

## Output Format

This README does not include a sample run, because no reference numbers have
been recorded. The program prints the following, in order:

1. A banner, the Metal device name, the vocabulary size, and the model and
   training configuration.
2. Train and validation token counts and the total parameter count.
3. The expected initial loss, `ln(vocab_size)`, which is about 4.17 for 65
   characters.
4. Every `logInterval` steps, a line of the form
   `step <n> | loss <x> | lr <x> | grad_norm <x> | <ms>/step | <s> elapsed`.
5. Every `evalInterval` steps, a `val_loss` line. A `*` marks a new best.
6. Every `generateInterval` steps, a generated text sample.
7. At the end, the total time, the final and best validation loss, and a
   500-token sample.

With nanoGPT's settings, the loss should fall from about 4.17 toward about 1.5
over 5000 steps. Your timings and final loss depend on your machine.

## Extending

To train on your own text:

1. Create a text file with your data.
2. Change `prepare_data.py` to read your file.
3. Adjust `GPT2Config` if you want a larger model.
4. Run training.

## References

- [nanoGPT](https://github.com/karpathy/nanoGPT), Andrej Karpathy's minimal GPT
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
