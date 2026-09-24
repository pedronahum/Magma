# Magma TPU Deployment Guide

> ⚠️ **Untested.** The maintainers have **not** run Magma on a TPU for
> `0.1.0-alpha.1`. The TPU code path loads `libtpu` as a PJRT plugin, the same way
> the verified CPU and CUDA paths load their plugins, but no part of it has been
> exercised on TPU hardware. Treat this guide as setup notes for an experiment.
> Please [report](https://github.com/pedronahum/Magma/issues) what works and what
> doesn't.

This guide covers setting up Magma on Google Cloud TPU VMs.

## Overview

Magma runs XLA programs through the PJRT (Portable JAX Runtime) plugin
interface. On TPU VMs the TPU runtime library (`libtpu.so`) is itself a PJRT
plugin. Magma loads it with `dlopen` at runtime, so you need no build flags.

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

# Download a Swift 6.0+ toolchain from swift.org (check swift.org for the latest
# release; Magma is tested on 6.0.3 and 6.3.3). Pick the aarch64 build if your
# TPU VM host is ARM.
wget https://download.swift.org/swift-6.0.3-release/ubuntu2204/swift-6.0.3-RELEASE/swift-6.0.3-RELEASE-ubuntu22.04.tar.gz
tar xzf swift-6.0.3-RELEASE-ubuntu22.04.tar.gz
sudo mv swift-6.0.3-RELEASE-ubuntu22.04 /opt/swift

# Add to PATH
echo 'export PATH=/opt/swift/usr/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
swift --version
```

### 4. Clone and Build Magma

```bash
# Clone the repository
git clone https://github.com/pedronahum/Magma.git
cd Magma

# Build (no flags: the TPU plugin is loaded at runtime)
swift build -c release
```

### 5. Point Magma at `libtpu`

Magma searches for the TPU plugin in this order. The first file that exists
wins.

1. `MAGMA_PJRT_PLUGIN_TPU`: the full path of the plugin file. When this is set,
   it is the only candidate.
2. `$MAGMA_XLA_PATH/pjrt_c_api_tpu_plugin.so`, `$MAGMA_XLA_PATH/libpjrt_c_api_tpu_plugin.so`
   and `$MAGMA_XLA_PATH/libtpu.so`.
3. `TPU_LIBRARY_PATH`, which is a file path, and then the standard locations
   `/usr/lib/libtpu.so`, `/usr/local/lib/libtpu.so` and `/lib/libtpu.so`.
4. The system directories `/opt/xla/lib`, `/usr/local/lib` and `/opt/magma/lib`.

Setting `MAGMA_XLA_PATH` (for example to find a CPU plugin) no longer hides
`libtpu`: the TPU-specific locations in step 3 are still searched. If nothing is
found, the error lists every path that was tried.

```bash
# Find libtpu on the VM (its location depends on the VM image and on whether it
# came from a pip package) and point Magma at it:
find / -name 'libtpu*.so' 2>/dev/null
export MAGMA_PJRT_PLUGIN_TPU=/path/to/libtpu.so

# Run on the TPU even for tensors created on the default (CPU) device
export MAGMA_DEFAULT_BACKEND=tpu
```

To see which file Magma will load, check `Backend.tpu.resolvedPluginPath` from
Swift.

## Using TPU in Magma

### Automatic TPU Detection

Magma automatically detects TPU availability:

```swift
import Magma

// Check if a TPU plugin was found (this only checks that the file exists)
if Backend.tpu.isAvailable {
    print("TPU plugin: \(Backend.tpu.resolvedPluginPath ?? "-")")
    TPUEnvironment.printInfo()
}

// The best available backend (TPU > Metal > GPU > CPU)
let backend = Backend.bestAvailable
print("Using backend: \(backend)")
```

### Creating a TPU Client

```swift

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

You can place work on the TPU in either of two ways:

- Set `MAGMA_DEFAULT_BACKEND=tpu`. Graphs then execute on the TPU even for
  tensors created on the default device.
- Create tensors and modules on `Device(backend: .tpu)` explicitly, using the
  `on:` / `device:` parameters or `to(device:)`.

```swift
import Magma

let tpu = Device(backend: .tpu, index: 0)

var model = nn.sequential {
    nn.Linear(inputSize: 784, outputSize: 256, device: tpu)
    nn.ReLU()
    nn.Linear(inputSize: 256, outputSize: 10, device: tpu)
}
model.eval()

let input = Tensor<Float>.randn([32, 784], on: tpu)
let output = model(input)
print("Output shape: \(output.shape)")
print(try output.fetchScalars().prefix(10))   // throws MaterializationError if execution fails
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
| `MAGMA_PJRT_PLUGIN_TPU` | Full path of the TPU plugin (`libtpu.so`); overrides every other location |
| `MAGMA_XLA_PATH` | Directory searched for `pjrt_c_api_tpu_plugin.so`, `libpjrt_c_api_tpu_plugin.so`, `libtpu.so` |
| `TPU_LIBRARY_PATH` | Full path of `libtpu.so`, searched after `MAGMA_XLA_PATH` |
| `MAGMA_DEFAULT_BACKEND` | Set to `tpu` to execute on the TPU regardless of the tensors' requested device |
| `MAGMA_ALLOW_CONCURRENT_ACCEL_CLIENTS` | Magma refuses a second concurrent GPU/TPU client in one process; set to `1` to override |
| `TPU_NAME` | TPU instance name (read by `TPUEnvironment`) |
| `TPU_CHIPS_PER_HOST_BOUNDS` | Chip topology, e.g. "2x2x1" (read by `TPUEnvironment`) |
| `TPU_HOST_BOUNDS` | Host topology (used by `TPUEnvironment.isMultiHost`) |
| `ACCELERATOR_TYPE` | TPU type, e.g. "v4-8" (read by `TPUEnvironment`) |

## Performance Tips

These are general TPU guidelines. They have not been benchmarked with Magma.

### 1. Use BFloat16

TPU matrix units are built for bfloat16. Magma can convert explicitly:

```swift
let weights = Tensor<Float>.randn([1024, 1024])
let bf16 = weights.toReducedPrecision()      // or weights.to(.bfloat16)
// MixedPrecision.autocast(inputs:computation:) runs a computation in bf16
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

If no plugin is found, reading a tensor fails with a `MaterializationError` at
stage `backendUnavailable`. The underlying error lists every path Magma searched
and names `MAGMA_XLA_PATH` and `MAGMA_PJRT_PLUGIN_TPU`.

**Solution**: find `libtpu.so` on the VM and point Magma at it:
```bash
find / -name 'libtpu*.so' 2>/dev/null
export MAGMA_PJRT_PLUGIN_TPU=/path/to/libtpu.so
```

If the file is found but fails to load, the error includes the `dlopen` or PJRT
message. The most likely cause is a PJRT C-API version mismatch between libtpu
and what Magma was built against. Magma is tested with PJRT C-API 0.108 (XLA
`9b635916`).

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

This is a complete training loop with real gradients. It uses the value-semantic
API: `modelValueWithGradient` differentiates the whole model and `Adam` updates
it. Run it with `MAGMA_DEFAULT_BACKEND=tpu` so the graphs execute on the TPU.
Like the rest of this guide, it has not been run on a TPU by the maintainers.

```swift
import Magma

guard Backend.tpu.isAvailable else {
    fatalError("No TPU plugin found. Set MAGMA_PJRT_PLUGIN_TPU or TPU_LIBRARY_PATH.")
}
TPUEnvironment.printInfo()

manualSeed(0)

// Synthetic data: replace with a real loader (e.g. MNIST, DataLoader).
let batchSize = 256, inputs = 784, classes = 10
let x = Tensor<Float>.randn([batchSize, inputs])
let labels = Tensor<Float>((0..<batchSize).map { Float($0 % classes) }, shape: [batchSize])
let y = Tensor<Float>.oneHot(labels, numClasses: classes)

func makeModel() -> some Layer {
    sequential {
        Linear(weight: Tensor<Float>.heUniform([inputs, 512]), bias: Tensor<Float>.zeros([512]))
        ReLU()
        Linear(weight: Tensor<Float>.heUniform([512, 256]), bias: Tensor<Float>.zeros([256]))
        ReLU()
        Linear(weight: Tensor<Float>.glorotUniform([256, classes]), bias: Tensor<Float>.zeros([classes]))
    }
}

// Cross-entropy written with differentiable primitives.
let n = Tensor<Float>.full([], Float(batchSize))
let eps = Tensor<Float>.full([], 1e-7)
let crossEntropy: @differentiable(reverse) (Tensor<Float>, Tensor<Float>) -> Tensor<Float> = { logits, target in
    -(target * (logits.softmax(dim: 1) + eps).log()).sum() / n
}

var model = makeModel()
var optimizer = Adam(learningRate: 0.001)

for step in 1...100 {
    let (loss, grad) = modelValueWithGradient(of: model, input: x, target: y, lossFn: crossEntropy)
    optimizer.update(&model, gradient: grad)     // materializes the new weights each step
    if step % 10 == 0 {
        print("step \(step): loss \(try loss.fetchItem())")
    }
}
print("Training complete!")
```

For `nn.*` models, compute gradients with `parameterGradients(of:loss:)` and
apply them with `optim.Adam(parameters:lr:).step(_:)`. See
[API.md: Training `nn` Modules](API.md#training-nn-modules).

## Next Steps

- See [ROADMAP.md](ROADMAP.md) for planned TPU and multi-device work.
- See [MULTI_DEVICE_ASSESSMENT.md](MULTI_DEVICE_ASSESSMENT.md) for the status of
  multi-device execution. It has been verified only on emulated CPU devices.
- If you try Magma on a TPU, please open an issue with your TPU type, libtpu
  version and results.
