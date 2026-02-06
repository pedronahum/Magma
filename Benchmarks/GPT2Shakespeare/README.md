# GPT-2 Shakespeare Example

Train a small GPT-2 model on Shakespeare text using Magma.

## Quick Start

```bash
# 1. Prepare data (download and tokenize Shakespeare)
python3 prepare_data.py

# 2. Run training
swift run -c release GPT2Shakespeare
```

## What This Demonstrates

- **GPT-2 Architecture**: Full transformer decoder with:
  - Token and position embeddings
  - Multi-head causal self-attention
  - Feed-forward MLP with GELU activation
  - Layer normalization (pre-norm style)

- **Training Features** (from Magma):
  - Gradient clipping (`optim.clipGradNorm`)
  - Learning rate scheduling (`WarmupCosineScheduler`)
  - Gradient accumulation (`optim.GradientAccumulator`)
  - Checkpointing (`TrainingState`)

- **Data Pipeline**:
  - Memory-mapped token files (`MemoryMappedTokenDataset`)
  - Character-level tokenization

## Model Configurations

| Config | Layers | Heads | Embed | Params |
|--------|--------|-------|-------|--------|
| test | 2 | 2 | 64 | ~50K |
| shakespeareTiny | 6 | 6 | 384 | ~10M |
| gpt2Small | 12 | 12 | 768 | ~124M |

## Files

- `main.swift` - GPT-2 model and training loop
- `prepare_data.py` - Data preparation script
- `data/` - Downloaded and tokenized data (created by prepare_data.py)

## Training Output

```
GPT-2 Shakespeare Training
==========================

Device: Apple M1

Model config:
  - Vocab size: 65
  - Block size: 64
  - Layers: 2
  - Heads: 2
  - Embedding dim: 64

Total parameters: 50000 (0.05M)

Iteration 1/100: loss = 4.1234
Iteration 2/100: loss = 3.9876
...
```

## Extending

To train on your own text:

1. Create a text file with your data
2. Modify `prepare_data.py` to point to your file
3. Adjust `GPT2Config` for larger models if needed
4. Run training

## References

- [nanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathy's minimal GPT
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
