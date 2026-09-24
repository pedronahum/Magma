// Magma - MNIST file handling tests (no network)
//
// The IDX parsers must throw, not trap, on truncated or corrupt files, and
// reject out-of-range labels. Decompression must not leave a partial file
// behind. All inputs here are synthetic bytes.

import Foundation
import Testing
@testable import Magma

@Suite("MNIST IDX parsing and caching")
struct MNISTParsingTests {

    private func be32(_ v: Int) -> [UInt8] {
        [UInt8(v >> 24 & 0xFF), UInt8(v >> 16 & 0xFF), UInt8(v >> 8 & 0xFF), UInt8(v & 0xFF)]
    }

    private func idx3(count: Int, rows: Int, cols: Int, pixels: [UInt8]) -> Data {
        Data(be32(0x0803) + be32(count) + be32(rows) + be32(cols) + pixels)
    }

    private func idx1(count: Int, labels: [UInt8]) -> Data {
        Data(be32(0x0801) + be32(count) + labels)
    }

    private func isInvalidFormat(_ error: any Error) -> Bool {
        if case MNISTError.invalidFormat = error { return true }
        return false
    }

    @Test("A well-formed IDX3 file parses to its header and pixels")
    func parsesValidIDX3() throws {
        let pixels: [UInt8] = Array(0..<12)
        let parsed = try MNIST.parseIDX3(idx3(count: 2, rows: 2, cols: 3, pixels: pixels))
        #expect(parsed.count == 2)
        #expect(parsed.rows == 2)
        #expect(parsed.cols == 3)
        #expect(parsed.pixels == pixels)
    }

    @Test("A truncated IDX3 payload throws instead of trapping")
    func truncatedIDX3Throws() {
        let data = idx3(count: 2, rows: 2, cols: 3, pixels: Array(repeating: 0, count: 11))
        #expect(performing: { try MNIST.parseIDX3(data) }, throws: isInvalidFormat)
    }

    @Test("Trailing bytes after the IDX3 payload are rejected")
    func oversizedIDX3Throws() {
        let data = idx3(count: 1, rows: 2, cols: 2, pixels: Array(repeating: 0, count: 5))
        #expect(performing: { try MNIST.parseIDX3(data) }, throws: isInvalidFormat)
    }

    @Test("A short header or wrong magic number throws")
    func badHeaderThrows() {
        #expect(performing: { try MNIST.parseIDX3(Data([0, 0, 8])) }, throws: isInvalidFormat)
        let wrongMagic = Data(be32(0x0801) + be32(1) + be32(1) + be32(1) + [7])
        #expect(performing: { try MNIST.parseIDX3(wrongMagic) }, throws: isInvalidFormat)
        #expect(performing: { try MNIST.parseIDX1(Data([0, 0]), numClasses: 10) }, throws: isInvalidFormat)
    }

    @Test("IDX1 labels parse; count mismatch and out-of-range labels throw")
    func idx1Validation() throws {
        #expect(try MNIST.parseIDX1(idx1(count: 3, labels: [0, 9, 4]), numClasses: 10) == [0, 9, 4])
        #expect(performing: {
            try MNIST.parseIDX1(idx1(count: 4, labels: [0, 9, 4]), numClasses: 10)
        }, throws: isInvalidFormat)
        #expect(performing: {
            try MNIST.parseIDX1(idx1(count: 2, labels: [3, 10]), numClasses: 10)
        }, throws: isInvalidFormat)
    }

    @Test("MAGMA_DATA_DIR overrides the default cache directory")
    func dataDirEnvironmentOverride() {
        #expect(MNIST.defaultDataDir(environment: ["MAGMA_DATA_DIR": "/data/sets"]) == "/data/sets/mnist")
        let fallback = MNIST.defaultDataDir(environment: [:])
        #expect(fallback.hasSuffix("/.magma/data/mnist"))
        #expect(MNIST.defaultDataDir(environment: ["MAGMA_DATA_DIR": ""]) == fallback)
    }

    @Test("gzip round trip, and a corrupt .gz throws without leaving a file")
    func gzipDecompression() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("magma-mnist-test-\(UUID().uuidString)").path
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        // Compress a small payload with the system gzip.
        let payload = Data(idx1(count: 3, labels: [1, 2, 3]))
        let raw = "\(dir)/labels"
        try payload.write(to: URL(fileURLWithPath: raw))
        let gzip = Process()
        gzip.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        gzip.arguments = ["gzip", "-f", raw]
        try gzip.run()
        gzip.waitUntilExit()
        try #require(gzip.terminationStatus == 0)

        let out = "\(dir)/labels.out"
        try MNIST.decompressGzip(from: "\(dir)/labels.gz", to: out)
        #expect(try Data(contentsOf: URL(fileURLWithPath: out)) == payload)

        // A corrupt archive fails cleanly: no destination, no partial file.
        let corrupt = "\(dir)/corrupt.gz"
        try Data([0x1F, 0x8B, 0x08, 0x00, 0x42, 0x42]).write(to: URL(fileURLWithPath: corrupt))
        let corruptOut = "\(dir)/corrupt.out"
        #expect(throws: MNISTError.self) {
            try MNIST.decompressGzip(from: corrupt, to: corruptOut)
        }
        #expect(!FileManager.default.fileExists(atPath: corruptOut))
        #expect(!FileManager.default.fileExists(atPath: corruptOut + ".partial"))
    }
}
