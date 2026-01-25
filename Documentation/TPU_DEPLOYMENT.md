# Magma TPU Deployment Guide

This guide covers deploying and running Magma on Google Cloud TPU VMs.

## Overview

Magma supports TPU acceleration through the PJRT (Portable JAX Runtime) interface. On TPU VMs, the TPU runtime library (`libtpu.so`) provides the PJRT plugin that enables XLA compilation and execution on TPU hardware.

## Prerequisites

- Google Cloud account with TPU quota
- `gcloud` CLI installed and configured
- SSH access to create TPU VMs

## TPU VM Setup

### 1. Create a TPU VM

```bash
# Set your project and zone
export PROJECT_ID="your-project-id"
export ZONE="us-central1-a"
export TPU_NAME="magma-tpu"

# Create a TPU v4-8 VM (8 TPU cores)
gcloud compute tpus tpu-vm create $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --accelerator-type=v4-8 \
    --version=tpu-ubuntu2204-base

# For TPU v3-8 (older generation, more availability)
gcloud compute tpus tpu-vm create $TPU_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --accelerator-type=v3-8 \
    --version=tpu-ubuntu2204-base
```

### 2. SSH into the TPU VM

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE
```

### 3. Install Swift on TPU VM

```bash
# Install Swift dependencies
sudo apt-get update
sudo apt-get install -y \
    binutils \
    git \
    gnupg2 \
    libc6-dev \
    libcurl4-openssl-dev \
    libedit2 \
    libgcc-11-dev \
    libpython3-dev \
    libsqlite3-0 \
    libstdc++-11-dev \
    libxml2-dev \
    libz3-dev \
    pkg-config \
    tzdata \
    unzip \
    zlib1g-dev

# Download Swift (check swift.org for latest version)
wget https://download.swift.org/swift-6.0-release/ubuntu2204/swift-6.0-RELEASE/swift-6.0-RELEASE-ubuntu22.04.tar.gz
tar xzf swift-6.0-RELEASE-ubuntu22.04.tar.gz
sudo mv swift-6.0-RELEASE-ubuntu22.04 /opt/swift

# Add to PATH
echo 'export PATH=/opt/swift/usr/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
swift --version
```

### 4. Clone and Build Magma

```bash
# Clone the repository
git clone https://github.com/your-org/Magma.git
cd Magma

# Build with TPU support
swift build -c release
```

## Using TPU in Magma

### Automatic TPU Detection

Magma automatically detects TPU availability:

```swift
import Torch
import XLARuntime

// Check if TPU is available
if Backend.tpu.isAvailable {
    print("TPU is available!")
    TPUEnvironment.printInfo()
}

// Use the best available backend (prefers TPU > GPU > CPU)
let backend = Backend.bestAvailable
print("Using backend: \(backend)")
```

### Creating a TPU Client

```swift
import XLARuntime

do {
    // Create a TPU client
    let client = try PJRTClient.create(backend: .tpu)
    print("TPU Platform: \(client.platformName)")
    print("TPU Devices: \(client.devices.count)")

    for device in client.devices {
        print("  - \(device.description)")
    }
} catch {
    print("Failed to create TPU client: \(error)")
}
```

### Running Models on TPU

```swift
import Torch

// Models automatically use the configured backend
let model = nn.sequential {
    nn.Linear(inputSize: 784, outputSize: 256)
    nn.ReLU()
    nn.Linear(inputSize: 256, outputSize: 10)
}

// Create tensors (will be placed on TPU when TPU backend is active)
let input = Tensor<Float>.randn([32, 784])
let output = model(input)
print("Output shape: \(output.shape)")
```

## TPU Types and Specifications

| TPU Type | Cores | HBM per Core | TFLOPs (bf16) | Recommended For |
|----------|-------|--------------|---------------|-----------------|
| v2-8     | 8     | 8 GB         | 180           | Small models, experimentation |
| v3-8     | 8     | 16 GB        | 420           | Medium models |
| v4-8     | 8     | 32 GB        | 275           | Large models, recommended |
| v5e-8    | 8     | 16 GB        | 197           | Inference, cost-effective |
| v5p-8    | 8     | 95 GB        | 459           | Largest models |

### TPU Pods (Multi-Host)

For larger workloads, TPU pods provide more cores:

| Configuration | Total Cores | Use Case |
|---------------|-------------|----------|
| v4-32         | 32          | Large batch training |
| v4-128        | 128         | Distributed training |
| v4-512        | 512         | Very large models |

## Environment Variables

Magma respects these TPU-related environment variables:

| Variable | Description |
|----------|-------------|
| `TPU_LIBRARY_PATH` | Custom path to libtpu.so |
| `TPU_NAME` | TPU instance name (set by Google Cloud) |
| `TPU_CHIPS_PER_HOST_BOUNDS` | Chip topology (e.g., "2x2x1") |
| `ACCELERATOR_TYPE` | TPU type (e.g., "v4-8") |

## Performance Tips

### 1. Use BFloat16

TPUs are optimized for bfloat16 operations:

```swift
// When available, prefer bfloat16 for TPU
let weights = Tensor<Float>.randn([1024, 1024])
// TPU will automatically use bf16 for matrix operations
```

### 2. Batch Size

TPUs perform best with larger batch sizes (powers of 2):

```swift
// Good batch sizes for TPU
let batchSizes = [128, 256, 512, 1024]
```

### 3. Avoid Host Transfers

Minimize data transfers between host and TPU:

```swift
// Bad: Many small transfers
for i in 0..<1000 {
    let x = Tensor<Float>.randn([1, 784])
    _ = model(x)  // Each call transfers data
}

// Good: Batch operations
let x = Tensor<Float>.randn([1000, 784])
_ = model(x)  // Single transfer
```

## Troubleshooting

### TPU Not Found

```
Error: Failed to load plugin '/usr/lib/libtpu.so'
```

**Solution**: Ensure you're on a TPU VM:
```bash
ls -la /usr/lib/libtpu.so
# Should exist on TPU VMs
```

### Out of Memory

```
Error: Resource exhausted: Out of memory
```

**Solution**: Reduce batch size or model size:
```swift
let smallerBatch = Tensor<Float>.randn([64, 784])  // Instead of 512
```

### TPU Lockout

If TPU becomes unresponsive:
```bash
# Reset TPU runtime
sudo systemctl restart tpu-runtime

# Or recreate the VM
gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE
gcloud compute tpus tpu-vm create $TPU_NAME --zone=$ZONE --accelerator-type=v4-8 --version=tpu-ubuntu2204-base
```

## Cost Optimization

### Preemptible TPUs

Use preemptible TPUs for 70-80% cost savings:

```bash
gcloud compute tpus tpu-vm create $TPU_NAME \
    --zone=$ZONE \
    --accelerator-type=v4-8 \
    --version=tpu-ubuntu2204-base \
    --preemptible
```

### Spot TPUs (Recommended)

Even cheaper than preemptible:

```bash
gcloud compute tpus tpu-vm create $TPU_NAME \
    --zone=$ZONE \
    --accelerator-type=v4-8 \
    --version=tpu-ubuntu2204-base \
    --spot
```

### Delete When Not in Use

TPUs are billed per hour. Delete when not needed:

```bash
gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE
```

## Example: Training on TPU

Complete example of training a model on TPU:

```swift
import Torch
import XLARuntime

// Check TPU availability
guard Backend.tpu.isAvailable else {
    fatalError("TPU not available. Run on a TPU VM.")
}

// Print environment info
TPUEnvironment.printInfo()

// Create TPU client
let client = try! PJRTClient.create(backend: .tpu)
print("Running on: \(client.platformName)")

// Define model
let model = nn.sequential {
    nn.Linear(inputSize: 784, outputSize: 512)
    nn.ReLU()
    nn.Linear(inputSize: 512, outputSize: 256)
    nn.ReLU()
    nn.Linear(inputSize: 256, outputSize: 10)
}

// Create optimizer
var optimizer = optim.Adam(parameters: model.parameters(), lr: 0.001)

// Training loop
let batchSize = 256  // Larger batches for TPU
let numEpochs = 10

for epoch in 0..<numEpochs {
    // Generate synthetic data (replace with real data loader)
    let input = Tensor<Float>.randn([batchSize, 784])
    let target = Tensor<Float>.zeros([batchSize, 10])

    // Forward pass
    let output = model(input)

    // Compute loss
    let loss = (output - target).pow(2.0).mean()

    // Backward pass (when autodiff is complete)
    // let grads = gradient(of: loss, wrt: model.parameters())

    // For now, use synthetic gradients
    let grads = model.parameters().map { p in
        Tensor<Float>.randn(p.shape) * Tensor<Float>.full([], 0.01, on: .default)
    }

    // Update weights
    optimizer.step(grads)

    print("Epoch \(epoch + 1)/\(numEpochs)")
}

print("Training complete!")
```

## Next Steps

- See [ROADMAP.md](ROADMAP.md) for upcoming TPU-specific features
- Check examples in `Examples/` for more TPU usage patterns
- For GPU support, see [GPU_DEPLOYMENT.md](GPU_DEPLOYMENT.md) (coming soon)
