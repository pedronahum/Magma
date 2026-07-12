// Magma - DistributedSampler tests
// Verify the data-sharding invariants a DDP data pipeline depends on: each
// replica gets an equal-sized shard, shards are disjoint and cover the dataset,
// and the shuffle is deterministic across replicas for a given seed+epoch.
// Pure Swift, no XLA.

import Testing
@testable import Magma

@Suite("Distributed Sampler Tests")
struct DistributedSamplerTests {

    private func shards(count: Int, numReplicas: Int, shuffle: Bool, epoch: Int = 0,
                        dropLast: Bool = false, seed: UInt64 = 0) -> [[Int]] {
        (0..<numReplicas).map { rank in
            DistributedSampler(count: count, numReplicas: numReplicas, rank: rank,
                               shuffle: shuffle, seed: seed, dropLast: dropLast).indices(epoch: epoch)
        }
    }

    @Test("shards are equal-sized")
    func equalSizes() {
        let s = shards(count: 100, numReplicas: 4, shuffle: true)
        #expect(s.map(\.count) == [25, 25, 25, 25])
    }

    @Test("shards are disjoint and cover the dataset (evenly divisible)")
    func disjointCovering() {
        let s = shards(count: 100, numReplicas: 4, shuffle: true)
        let union = Set(s.flatMap { $0 })
        #expect(union == Set(0..<100))                 // full coverage
        #expect(s.flatMap { $0 }.count == 100)         // no overlap (no repeats)
    }

    @Test("uneven dataset pads so every replica sees the same count")
    func unevenPadding() {
        // 10 samples / 4 replicas -> numSamples = 3 each, total 12 (2 padded).
        let sampler = DistributedSampler(count: 10, numReplicas: 4, rank: 0)
        #expect(sampler.numSamples == 3)
        #expect(sampler.totalSize == 12)
        let s = shards(count: 10, numReplicas: 4, shuffle: false)
        #expect(s.allSatisfy { $0.count == 3 })
        // Every original index is covered at least once.
        #expect(Set(s.flatMap { $0 }) == Set(0..<10))
    }

    @Test("dropLast truncates instead of padding")
    func dropLastTruncates() {
        // 10 / 4 -> floor = 2 each, total 8.
        let sampler = DistributedSampler(count: 10, numReplicas: 4, rank: 0, dropLast: true)
        #expect(sampler.numSamples == 2)
        let s = shards(count: 10, numReplicas: 4, shuffle: true, dropLast: true)
        #expect(s.allSatisfy { $0.count == 2 })
        #expect(s.flatMap { $0 }.count == 8)           // 8 distinct, 2 dropped
        #expect(Set(s.flatMap { $0 }).count == 8)
    }

    @Test("shuffle is deterministic for a given seed+epoch, varies by epoch")
    func deterministicShuffle() {
        let a = DistributedSampler(count: 50, numReplicas: 2, rank: 0, shuffle: true, seed: 7)
        let b = DistributedSampler(count: 50, numReplicas: 2, rank: 0, shuffle: true, seed: 7)
        #expect(a.indices(epoch: 0) == b.indices(epoch: 0))     // reproducible
        #expect(a.indices(epoch: 0) != a.indices(epoch: 1))     // epoch changes order
    }

    @Test("all replicas share one permutation (disjoint even when shuffled)")
    func sharedPermutation() {
        let s = shards(count: 40, numReplicas: 4, shuffle: true, seed: 123)
        // Pairwise disjoint.
        for i in 0..<4 {
            for j in (i + 1)..<4 {
                #expect(Set(s[i]).isDisjoint(with: Set(s[j])))
            }
        }
        #expect(Set(s.flatMap { $0 }) == Set(0..<40))
    }

    @Test("per-replica batch size splits the global batch")
    func perReplicaBatch() {
        #expect(DistributedSampler.perReplicaBatchSize(globalBatchSize: 128, numReplicas: 4) == 32)
        #expect(DistributedSampler.perReplicaBatchSize(globalBatchSize: 10, numReplicas: 3) == 3)
        #expect(DistributedSampler.perReplicaBatchSize(globalBatchSize: 1, numReplicas: 4) == 1)
    }
}

@Suite("DataLoader Distributed Integration")
struct DataLoaderDistributedTests {

    @Test("a distributed DataLoader iterates only its replica's shard")
    func loaderUsesShard() {
        let n = 20
        let inputs = Tensor<Float>.arange(n).reshape([n, 1])
        let targets = Tensor<Float>.arange(n).reshape([n, 1])
        let dataset = TensorDataset(inputs: inputs, targets: targets)

        // 2 replicas, per-replica batch size 5 -> each replica sees 10 samples
        // = 2 batches.
        let s0 = DistributedSampler(count: n, numReplicas: 2, rank: 0, shuffle: false)
        let s1 = DistributedSampler(count: n, numReplicas: 2, rank: 1, shuffle: false)
        let l0 = DataLoader(dataset: dataset, batchSize: 5, sampler: s0)
        let l1 = DataLoader(dataset: dataset, batchSize: 5, sampler: s1)

        #expect(l0.count == 2)
        #expect(l1.count == 2)

        func totalRows(_ loader: DataLoader<TensorDataset>) -> Int {
            loader.reduce(0) { $0 + $1.input.shape[0] }
        }
        #expect(totalRows(l0) == 10)
        #expect(totalRows(l1) == 10)
    }
}
