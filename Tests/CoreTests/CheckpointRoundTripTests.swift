// Magma - Checkpoint round-trips by value, buffers, format compatibility and
// corrupt-file handling.
// Module.save/load and BinaryCheckpoint used to drop buffers() (BatchNorm
// running statistics), so a reloaded model ran inference with the initial
// statistics; the existing tests only compared shapes. Corrupt or truncated
// files used to trap (unchecked Data ranges / indices) instead of throwing.
//
// .serialized: materialization drives the PJRT backend.

import Foundation
import Testing
@testable import Magma

private func tempURL(_ ext: String) -> URL {
    FileManager.default.temporaryDirectory
        .appendingPathComponent("magma-ckpt-\(UUID().uuidString).\(ext)")
}

private func fill(_ shape: [Int], _ scale: Float = 0.1, _ bias: Float = -0.3, mod: Int = 7) -> Tensor<Float> {
    let n = shape.reduce(1, *)
    return Tensor<Float>((0..<n).map { Float($0 % mod) * scale + bias }, shape: shape)
}

private func makeModel() -> nn.Sequential {
    nn.sequential {
        nn.Linear(inputSize: 3, outputSize: 4)
        nn.BatchNorm1d(numFeatures: 4)
    }
}

/// A model whose BatchNorm running stats have moved away from their init.
private func trainedModelInEval() -> nn.Sequential {
    var model = makeModel()
    for step in 0..<3 {
        _ = model(fill([6, 3], 0.3 + Float(step) * 0.1, -0.5 + Float(step)))
    }
    model.eval()
    return model
}

private func values(_ params: [Parameter]) -> [[Float]] {
    params.map { $0.value.scalars() }
}

@Suite("Checkpoint round-trips (materialized)", .serialized)
struct CheckpointRoundTripTests {

    @Test("JSON save/load restores parameter and buffer values and eval output")
    func jsonRoundTripWithBuffers() throws {
        let model = trainedModelInEval()
        let x = fill([5, 3], 0.2, -0.4)
        let expected = model(x).scalars()
        // Running stats really moved, so dropping them would change the output.
        #expect(model.buffers()[0].value.scalars() != [0, 0, 0, 0])

        let url = tempURL("json")
        defer { try? FileManager.default.removeItem(at: url) }
        try model.save(to: url)

        var loaded = makeModel()
        loaded.eval()
        #expect(loaded(x).scalars() != expected)
        try loaded.load(from: url)

        #expect(values(loaded.parameters()) == values(model.parameters()))
        #expect(values(loaded.buffers()) == values(model.buffers()))
        #expect(loaded(x).scalars() == expected)
    }

    @Test("BinaryCheckpoint save/load restores parameter and buffer values and eval output")
    func binaryRoundTripWithBuffers() throws {
        let model = trainedModelInEval()
        let x = fill([5, 3], 0.2, -0.4)
        let expected = model(x).scalars()

        let url = tempURL("bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try BinaryCheckpoint.save(model, to: url)

        var loaded = makeModel()
        loaded.eval()
        try BinaryCheckpoint.load(&loaded, from: url)

        #expect(values(loaded.parameters()) == values(model.parameters()))
        #expect(values(loaded.buffers()) == values(model.buffers()))
        #expect(loaded(x).scalars() == expected)
    }

    @Test("BatchNorm2d running stats survive a round-trip")
    func batchNorm2dRoundTrip() throws {
        var bn = nn.BatchNorm2d(numFeatures: 2)
        _ = bn(fill([2, 3, 3, 2], 0.5, -1.0, mod: 11))
        bn.eval()
        let x = fill([1, 2, 2, 2], 0.3, -0.2)
        let expected = bn(x).scalars()

        let url = tempURL("bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try BinaryCheckpoint.save(bn, to: url)
        var loaded = nn.BatchNorm2d(numFeatures: 2)
        loaded.eval()
        try BinaryCheckpoint.load(&loaded, from: url)
        #expect(loaded(x).scalars() == expected)
    }

    // MARK: Older formats

    @Test("version-1 JSON checkpoints (no buffers) still load")
    func loadsVersion1JSON() throws {
        var linear = nn.Linear(inputSize: 2, outputSize: 1)
        let wShape = linear.weight.shape
        let json = """
        {"version": 1, "timestamp": 0, "parameters": [
          {"name": "weight", "index": 0, "shape": \(wShape), "values": [1.5, -2.0]},
          {"name": "bias", "index": 1, "shape": [1], "values": [0.25]}
        ]}
        """
        let url = tempURL("json")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(json.utf8).write(to: url)

        try linear.load(from: url)
        #expect(linear.weight.value.scalars() == [1.5, -2.0])
        #expect(linear.bias.value.scalars() == [0.25])
    }

    @Test("version-1 binary checkpoints (no buffer section) still load")
    func loadsVersion1Binary() throws {
        var linear = nn.Linear(inputSize: 2, outputSize: 1)
        var data = Data("STCHKPT\0".utf8)
        func u32(_ v: Int) { var le = UInt32(v).littleEndian; data.append(Data(bytes: &le, count: 4)) }
        func f32(_ v: [Float]) { v.withUnsafeBufferPointer { data.append($0) } }
        u32(1); u32(2)                                           // version 1, 2 params
        u32(1); data.append(Data("w".utf8))                      // name
        u32(linear.weight.shape.count); linear.weight.shape.forEach { u32($0) }
        f32([0.5, 0.75])
        u32(1); data.append(Data("b".utf8)); u32(1); u32(1); f32([-1])

        let url = tempURL("bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)

        try BinaryCheckpoint.load(&linear, from: url)
        #expect(linear.weight.value.scalars() == [0.5, 0.75])
        #expect(linear.bias.value.scalars() == [-1])
    }

    // MARK: Corrupt / truncated files

    @Test("truncated binary checkpoints throw and leave the module unchanged")
    func truncatedBinaryThrows() throws {
        let model = trainedModelInEval()
        let url = tempURL("bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try BinaryCheckpoint.save(model, to: url)
        let full = try Data(contentsOf: url)

        var target = makeModel()
        let before = values(target.parameters()) + values(target.buffers())
        for length in [0, 4, 8, 12, 17, full.count / 2, full.count - 1] {
            try full.prefix(length).write(to: url)
            #expect(throws: CheckpointError.self, "prefix of \(length) bytes") {
                try BinaryCheckpoint.load(&target, from: url)
            }
        }
        // Trailing garbage is rejected too.
        try (full + Data([1, 2, 3])).write(to: url)
        #expect(throws: CheckpointError.self) { try BinaryCheckpoint.load(&target, from: url) }

        #expect(values(target.parameters()) + values(target.buffers()) == before)
    }

    @Test("a binary checkpoint with a garbage shape throws instead of allocating/trapping")
    func garbageShapeThrows() throws {
        var linear = nn.Linear(inputSize: 2, outputSize: 1)
        var data = Data("STCHKPT\0".utf8)
        func u32(_ v: UInt32) { var le = v.littleEndian; data.append(Data(bytes: &le, count: 4)) }
        u32(2); u32(2)                       // version 2, 2 params
        u32(1); data.append(Data("w".utf8))
        u32(0xFFFF_FFFF)                     // absurd rank
        let url = tempURL("bin")
        defer { try? FileManager.default.removeItem(at: url) }
        try data.write(to: url)
        #expect(throws: CheckpointError.self) { try BinaryCheckpoint.load(&linear, from: url) }
    }

    @Test("JSON checkpoints with a bad index or value count throw")
    func corruptJSONThrows() throws {
        var linear = nn.Linear(inputSize: 2, outputSize: 1)
        let wShape = linear.weight.shape
        let url = tempURL("json")
        defer { try? FileManager.default.removeItem(at: url) }

        let badIndex = """
        {"version": 2, "timestamp": 0, "buffers": [], "parameters": [
          {"index": 7, "shape": \(wShape), "values": [1, 2]},
          {"index": 1, "shape": [1], "values": [0]}
        ]}
        """
        try Data(badIndex.utf8).write(to: url)
        #expect(throws: CheckpointError.self) { try linear.load(from: url) }

        let shortValues = """
        {"version": 2, "timestamp": 0, "buffers": [], "parameters": [
          {"index": 0, "shape": \(wShape), "values": [1]},
          {"index": 1, "shape": [1], "values": [0]}
        ]}
        """
        try Data(shortValues.utf8).write(to: url)
        #expect(throws: CheckpointError.self) { try linear.load(from: url) }

        let futureVersion = """
        {"version": 99, "timestamp": 0, "parameters": []}
        """
        try Data(futureVersion.utf8).write(to: url)
        #expect(throws: CheckpointError.self) { try linear.load(from: url) }
    }
}
