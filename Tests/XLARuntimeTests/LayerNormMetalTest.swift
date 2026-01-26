// LayerNormMetalTest.swift
// Test LayerNorm specifically on Metal backend

import Foundation
import Testing
@testable import XLARuntime
@testable import LazyTensor
@testable import StableHLO
@testable import Magma

#if os(macOS) && canImport(MetalHLO)
import MetalHLO

@Suite("LayerNorm Metal Tests")
struct LayerNormMetalTests {

    @Test("Mean reduction on Metal")
    func meanReductionMetal() throws {
        let metalDevice = Device(backend: .metal, index: 0)

        // Test simple mean reduction
        let x = Tensor<Float>.randn([2, 4], on: metalDevice)
        x.markForMaterialization()
        LazyTensorBarrier(on: metalDevice)

        let mean = x.mean(dims: [1], keepDims: true)
        LazyTensorBarrier(on: metalDevice)

        print("Mean shape: \(mean.shape)")
        #expect(mean.shape == [2, 1])
        let _ = mean.scalars().first
    }

    @Test("Basic broadcast on Metal")
    func basicBroadcastMetal() throws {
        let metalDevice = Device(backend: .metal, index: 0)

        // Test simple broadcast: [4] -> [2, 4]
        let a = Tensor<Float>.ones([4], on: metalDevice)
        let b = Tensor<Float>.randn([2, 4], on: metalDevice)

        a.markForMaterialization()
        b.markForMaterialization()
        LazyTensorBarrier(on: metalDevice)

        let c = a.broadcast(to: [2, 4]) * b
        LazyTensorBarrier(on: metalDevice)

        #expect(c.shape == [2, 4])
        let _ = c.scalars().first
    }

    @Test("Mean + broadcast pattern on Metal")
    func meanBroadcastMetal() throws {
        let metalDevice = Device(backend: .metal, index: 0)

        let x = Tensor<Float>.randn([2, 4], on: metalDevice)
        x.markForMaterialization()
        LazyTensorBarrier(on: metalDevice)

        // Mean with keepDims
        let mean = x.mean(dims: [1], keepDims: true)
        print("Mean shape: \(mean.shape)")

        // Subtract mean (should broadcast automatically)
        let centered = x - mean.broadcast(to: x.shape)
        LazyTensorBarrier(on: metalDevice)

        print("Centered shape: \(centered.shape)")
        #expect(centered.shape == [2, 4])
        let _ = centered.scalars().first
    }

    @Test("Simple LayerNorm manual on Metal")
    func simpleLayerNormManual() throws {
        let metalDevice = Device(backend: .metal, index: 0)

        let x = Tensor<Float>.randn([2, 4], on: metalDevice)
        let gamma = Tensor<Float>.ones([4], on: metalDevice)
        let beta = Tensor<Float>.zeros([4], on: metalDevice)
        let eps: Float = 1e-5

        x.markForMaterialization()
        gamma.markForMaterialization()
        beta.markForMaterialization()
        LazyTensorBarrier(on: metalDevice)

        // Compute mean
        let mean = x.mean(dims: [1], keepDims: true)

        // Compute variance
        let centered = x - mean.broadcast(to: x.shape)
        let variance = (centered * centered).mean(dims: [1], keepDims: true)

        // Normalize
        let epsT = Tensor<Float>.full(variance.shape, eps, on: metalDevice)
        let stddev = (variance + epsT).sqrt()
        let normalized = centered / stddev.broadcast(to: x.shape)

        // Scale and shift
        let scaled = normalized * gamma.broadcast(to: x.shape)
        let output = scaled + beta.broadcast(to: x.shape)

        LazyTensorBarrier(on: metalDevice)

        print("LayerNorm output shape: \(output.shape)")
        #expect(output.shape == [2, 4])
        let _ = output.scalars().first
    }

    @Test("nn.LayerNorm on Metal")
    func nnLayerNormMetal() throws {
        let metalDevice = Device(backend: .metal, index: 0)

        let ln = nn.LayerNorm(4, device: metalDevice)
        let x = Tensor<Float>.randn([2, 4], on: metalDevice)

        x.markForMaterialization()
        LazyTensorBarrier(on: metalDevice)

        let y = ln(x)
        LazyTensorBarrier(on: metalDevice)

        print("nn.LayerNorm output shape: \(y.shape)")
        #expect(y.shape == [2, 4])
        let _ = y.scalars().first
    }

    @Test("nn.LayerNorm 3D on Metal")
    func nnLayerNorm3DMetal() throws {
        let metalDevice = Device(backend: .metal, index: 0)

        // 3D: [batch, seq, features]
        let ln = nn.LayerNorm(64, device: metalDevice)
        let x = Tensor<Float>.randn([8, 16, 64], on: metalDevice)

        x.markForMaterialization()
        LazyTensorBarrier(on: metalDevice)

        let y = ln(x)
        LazyTensorBarrier(on: metalDevice)

        print("nn.LayerNorm 3D output shape: \(y.shape)")
        #expect(y.shape == [8, 16, 64])
        let _ = y.scalars().first
    }

    @Test("Attention pattern Q @ K^T on Metal (small)")
    func attentionPatternMetalSmall() throws {
        let metalDevice = Device(backend: .metal, index: 0)

        // Small attention pattern: [batch, heads, seq, headDim]
        let batch = 2
        let heads = 2
        let seqLen = 4
        let headDim = 3

        let q = Tensor<Float>.randn([batch, heads, seqLen, headDim], on: metalDevice)
        let k = Tensor<Float>.randn([batch, heads, seqLen, headDim], on: metalDevice)

        q.markForMaterialization()
        k.markForMaterialization()
        LazyTensorBarrier(on: metalDevice)

        // K^T: swap last two dims [batch, heads, headDim, seqLen]
        let kT = k.transposeLastTwo()
        print("Q shape: \(q.shape)")
        print("K shape: \(k.shape)")
        print("K^T shape: \(kT.shape)")
        #expect(kT.shape == [batch, heads, headDim, seqLen])

        // Q @ K^T -> [batch, heads, seqLen, seqLen]
        let scores = q.batchedMatmul(kT)
        LazyTensorBarrier(on: metalDevice)

        print("Scores shape: \(scores.shape)")
        #expect(scores.shape == [batch, heads, seqLen, seqLen])
        let _ = scores.scalars().first
    }

    @Test("Attention pattern Q @ K^T on Metal (benchmark size)")
    func attentionPatternMetalLarge() throws {
        let metalDevice = Device(backend: .metal, index: 0)

        // Benchmark-size attention pattern: [batch, heads, seq, headDim]
        let batch = 4
        let heads = 8
        let seqLen = 128
        let headDim = 64

        let q = Tensor<Float>.randn([batch, heads, seqLen, headDim], on: metalDevice)
        let k = Tensor<Float>.randn([batch, heads, seqLen, headDim], on: metalDevice)

        q.markForMaterialization()
        k.markForMaterialization()
        LazyTensorBarrier(on: metalDevice)

        // K^T: swap last two dims [batch, heads, headDim, seqLen]
        let kT = k.transposeLastTwo()
        print("Q shape: \(q.shape)")
        print("K shape: \(k.shape)")
        print("K^T shape: \(kT.shape)")
        #expect(kT.shape == [batch, heads, headDim, seqLen])

        // Q @ K^T -> [batch, heads, seqLen, seqLen]
        let scores = q.batchedMatmul(kT)
        LazyTensorBarrier(on: metalDevice)

        print("Scores shape: \(scores.shape)")
        #expect(scores.shape == [batch, heads, seqLen, seqLen])
        let _ = scores.scalars().first
    }

    @Test("Full attention pattern on Metal (small)")
    func fullAttentionPatternMetalSmall() throws {
        let metalDevice = Device(backend: .metal, index: 0)

        let batch = 2
        let heads = 2
        let seqLen = 4
        let headDim = 8
        let scale = 1.0 / sqrt(Float(headDim))

        let q = Tensor<Float>.randn([batch, heads, seqLen, headDim], on: metalDevice)
        let k = Tensor<Float>.randn([batch, heads, seqLen, headDim], on: metalDevice)
        let v = Tensor<Float>.randn([batch, heads, seqLen, headDim], on: metalDevice)

        q.markForMaterialization()
        k.markForMaterialization()
        v.markForMaterialization()
        LazyTensorBarrier(on: metalDevice)

        // Full attention: (Q @ K^T / sqrt(d)) @ V with softmax
        let kT = k.transpose(-1, -2)
        let scores = q.batchedMatmul(kT) * Tensor<Float>.full([], scale, on: metalDevice)
        let attnWeights = scores.softmax(dim: -1)
        let output = attnWeights.batchedMatmul(v)
        LazyTensorBarrier(on: metalDevice)

        print("Output shape: \(output.shape)")
        #expect(output.shape == [batch, heads, seqLen, headDim])
        let _ = output.scalars().first
    }
}

#endif
