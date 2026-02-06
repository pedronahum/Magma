#!/usr/bin/env python3
"""
Prepare Shakespeare data for GPT-2 training.

This script downloads the tiny Shakespeare dataset and creates:
1. train.bin - Training tokens (90%)
2. val.bin - Validation tokens (10%)
3. meta.json - Vocabulary mapping

Usage:
    python prepare_data.py

The output files can be loaded in Magma using MemoryMappedTokenDataset.
"""

import os
import json
import numpy as np
import urllib.request

# Configuration
DATA_DIR = "data"
TRAIN_SPLIT = 0.9

def download_shakespeare():
    """Download tiny Shakespeare dataset."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filepath = os.path.join(DATA_DIR, "input.txt")

    if os.path.exists(filepath):
        print(f"Using cached {filepath}")
        with open(filepath, 'r') as f:
            return f.read()

    print(f"Downloading Shakespeare from {url}...")
    os.makedirs(DATA_DIR, exist_ok=True)

    with urllib.request.urlopen(url) as response:
        text = response.read().decode('utf-8')

    with open(filepath, 'w') as f:
        f.write(text)

    print(f"Downloaded {len(text)} characters")
    return text

def create_vocab(text):
    """Create character-level vocabulary."""
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    print(f"Vocabulary size: {vocab_size}")
    print(f"Characters: {''.join(chars[:50])}...")

    return stoi, itos, vocab_size

def encode(text, stoi):
    """Encode text to token IDs."""
    return [stoi[ch] for ch in text]

def decode(ids, itos):
    """Decode token IDs to text."""
    return ''.join([itos[i] for i in ids])

def main():
    print("=" * 60)
    print("Shakespeare Data Preparation")
    print("=" * 60)
    print()

    # Download data
    text = download_shakespeare()
    print(f"Total characters: {len(text)}")
    print()

    # Create vocabulary
    stoi, itos, vocab_size = create_vocab(text)
    print()

    # Encode entire text
    print("Encoding text...")
    data = encode(text, stoi)
    data = np.array(data, dtype=np.uint16)
    print(f"Total tokens: {len(data)}")
    print()

    # Split into train/val
    n = int(len(data) * TRAIN_SPLIT)
    train_data = data[:n]
    val_data = data[n:]

    print(f"Train tokens: {len(train_data)}")
    print(f"Val tokens: {len(val_data)}")
    print()

    # Save binary files
    train_path = os.path.join(DATA_DIR, "train.bin")
    val_path = os.path.join(DATA_DIR, "val.bin")

    train_data.tofile(train_path)
    val_data.tofile(val_path)

    print(f"Saved {train_path} ({os.path.getsize(train_path)} bytes)")
    print(f"Saved {val_path} ({os.path.getsize(val_path)} bytes)")
    print()

    # Save metadata
    meta = {
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},  # JSON needs string keys
    }
    meta_path = os.path.join(DATA_DIR, "meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {meta_path}")
    print()

    # Verify
    print("Verifying...")
    loaded = np.memmap(train_path, dtype=np.uint16, mode='r')
    sample = decode(loaded[:100].tolist(), itos)
    print(f"First 100 chars: {repr(sample)}")
    print()

    print("Done! You can now run GPT2Shakespeare with:")
    print("  swift run GPT2Shakespeare")

if __name__ == "__main__":
    main()
