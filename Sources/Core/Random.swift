// Magma - Host-side random number generation
//
// Every random value Magma draws on the host (Tensor.randn / uniform, the
// weight initializers, nn.Dropout masks, DataLoader / SimpleBatchLoader
// shuffles and the random data transforms) comes from one process-wide
// generator. Unseeded it is backed by the system RNG; `manualSeed(_:)` switches
// it to a deterministic SplitMix64 stream so runs can be reproduced.

import Foundation

/// Seeds Magma's global host-side random number generator.
///
/// After this call every host-side random draw is deterministic: the same seed
/// followed by the same sequence of calls produces the same values. This covers
/// `Tensor.randn`, `Tensor.uniform`, the weight initializers (`truncatedNormal`,
/// `randomUniform`, `glorotUniform`, `heNormal`, ...), `nn.Dropout` masks,
/// `DataLoader` / `SimpleBatchLoader` shuffling and the random data transforms.
/// Call it again with the same seed to restart the stream.
///
/// Until `manualSeed(_:)` is called, the generator draws from the system RNG, so
/// unseeded programs stay nondeterministic.
///
/// ```swift
/// manualSeed(42)
/// let model = nn.Linear(inputSize: 4, outputSize: 2)   // same weights every run
/// ```
///
/// - Note: Device-side RNG ops (`Tensor.randnDevice`, `Tensor.randDevice`,
///   `randn(_:useDeviceRNG: true)` and the Gumbel noise used by `multinomial`) run
///   inside XLA and are **not** controlled by this seed. `DistributedSampler`
///   keeps its own explicit `seed:` so that replicas agree independently of
///   the global stream.
/// - Note: The generator is shared by all threads. Draws from concurrent tasks
///   interleave in an unspecified order, so determinism is only guaranteed when
///   random values are drawn from one thread (or in a fixed order).
///
/// - Parameter seed: The seed for the host random stream.
public func manualSeed(_ seed: UInt64) {
    HostRandom.seed(seed)
}

/// The process-wide host random stream behind `manualSeed(_:)`.
enum HostRandom {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var generator = HostGenerator()

    /// Replaces the stream with a deterministic one started from `seed`.
    static func seed(_ seed: UInt64) {
        lock.lock()
        generator = HostGenerator(seeded: SeededGenerator(seed: seed))
        lock.unlock()
    }

    /// Runs `body` with exclusive access to the global generator.
    ///
    /// Hold the generator for a whole batch of draws (e.g. all elements of one
    /// tensor) so that concurrent callers cannot interleave inside it.
    static func withGenerator<R>(_ body: (inout HostGenerator) throws -> R) rethrows -> R {
        lock.lock()
        defer { lock.unlock() }
        return try body(&generator)
    }
}

/// The global host generator: seeded SplitMix64 after `manualSeed(_:)`, the
/// system RNG before.
struct HostGenerator: RandomNumberGenerator {
    private var seeded: SeededGenerator?
    private var system = SystemRandomNumberGenerator()

    init(seeded: SeededGenerator? = nil) {
        self.seeded = seeded
    }

    mutating func next() -> UInt64 {
        if seeded != nil {
            return seeded!.next()
        }
        return system.next()
    }
}

/// Draws `count` standard-normal samples with the Box-Muller transform.
func hostNormalSamples(count: Int, using rng: inout HostGenerator) -> [Float] {
    var values: [Float] = []
    values.reserveCapacity(count)
    while values.count < count {
        // u1 in (0, 1] keeps log(u1) finite; u2 in [0, 1).
        let u1 = Float.random(in: Float.leastNormalMagnitude...1, using: &rng)
        let u2 = Float.random(in: 0..<1, using: &rng)
        let mag = Foundation.sqrt(-2.0 * Foundation.log(u1))
        values.append(mag * Foundation.cos(2.0 * Float.pi * u2))
        if values.count < count {
            values.append(mag * Foundation.sin(2.0 * Float.pi * u2))
        }
    }
    return values
}

/// One sample from U[0, 1) drawn from the global host stream.
func hostUniformSample() -> Float {
    HostRandom.withGenerator { Float.random(in: 0..<1, using: &$0) }
}
