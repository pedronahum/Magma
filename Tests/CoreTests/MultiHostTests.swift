// Magma - Multi-host topology & data-sharding tests
// The pure-Swift, testable part of multi-host: process/device topology and
// globally-disjoint data sharding across a simulated cluster. (The coordination
// service itself needs a real multi-host runtime and is out of scope here.)

import Testing
@testable import Magma

@Suite("Multi-Host Tests")
struct MultiHostTests {

    @Test("global topology from per-process config")
    func topology() {
        // 3 processes x 2 local devices = 6 global devices.
        let c0 = MultiHostConfig(processIndex: 0, numProcesses: 3, localDeviceCount: 2)
        let c1 = MultiHostConfig(processIndex: 1, numProcesses: 3, localDeviceCount: 2)
        let c2 = MultiHostConfig(processIndex: 2, numProcesses: 3, localDeviceCount: 2)

        #expect(c0.globalDeviceCount == 6)
        #expect(c0.isCoordinator)
        #expect(!c1.isCoordinator)

        // Contiguous global-index blocks, no overlap.
        #expect(c0.localGlobalIndices == [0, 1])
        #expect(c1.localGlobalIndices == [2, 3])
        #expect(c2.localGlobalIndices == [4, 5])
        #expect(c1.globalDeviceIndex(localDevice: 1) == 3)
    }

    @Test("multi-host data sharding is globally disjoint and covering")
    func globalSharding() {
        // 2 processes x 2 local devices = 4 global ranks over a 40-sample set.
        let count = 40
        let procs = 2, localDevs = 2
        var allShards: [[Int]] = []
        for p in 0..<procs {
            let cfg = MultiHostConfig(processIndex: p, numProcesses: procs, localDeviceCount: localDevs)
            for d in 0..<localDevs {
                let sampler = DistributedSampler.multiHost(
                    count: count, config: cfg, localDevice: d, shuffle: true, seed: 99)
                allShards.append(sampler.indices(epoch: 0))
            }
        }

        #expect(allShards.count == 4)
        #expect(allShards.allSatisfy { $0.count == 10 })     // 40 / 4

        // Pairwise disjoint across the whole cluster.
        for i in 0..<allShards.count {
            for j in (i + 1)..<allShards.count {
                #expect(Set(allShards[i]).isDisjoint(with: Set(allShards[j])))
            }
        }
        // Together they cover the dataset.
        #expect(Set(allShards.flatMap { $0 }) == Set(0..<count))
    }

    @Test("shared seed gives the same permutation across processes")
    func sharedPermutation() {
        // Two processes must agree on the base permutation (same seed+epoch) so
        // their strided shards don't overlap.
        let a = DistributedSampler.multiHost(
            count: 20, config: MultiHostConfig(processIndex: 0, numProcesses: 2, localDeviceCount: 1),
            localDevice: 0, shuffle: true, seed: 7)
        let b = DistributedSampler.multiHost(
            count: 20, config: MultiHostConfig(processIndex: 1, numProcesses: 2, localDeviceCount: 1),
            localDevice: 0, shuffle: true, seed: 7)
        #expect(Set(a.indices()).isDisjoint(with: Set(b.indices())))
        #expect(Set(a.indices() + b.indices()) == Set(0..<20))
    }
}
