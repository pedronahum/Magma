// Magma - Multi-host topology & data sharding
//
// The pure-Swift, testable part of multi-host distribution: the process/device
// topology and global data sharding across an entire cluster. Each process owns
// `localDeviceCount` devices; a global device index spans all processes.
//
// NOT included here (requires a real multi-host runtime, not testable on a single
// box): PJRT distributed initialization — a coordination service (gRPC) with a
// KV store for rendezvous, global device-id assignment, and the NCCL unique-id
// exchange on GPU. Those are passed to `PJRT_Client_Create` (create-options /
// key-value callbacks) when launching one process per host; `MultiHostConfig` is
// the configuration such an init consumes. Single-host multi-device (emulated CPU
// or one multi-GPU box) needs none of it and is already covered.

import Foundation

/// Topology of a multi-host (multi-process) distributed job.
public struct MultiHostConfig: Sendable, Equatable {
    /// This process's index in `[0, numProcesses)`.
    public let processIndex: Int
    /// Total number of processes (hosts) in the job.
    public let numProcesses: Int
    /// Number of devices this process manages.
    public let localDeviceCount: Int
    /// Address of the coordination service (host:port), used by the distributed
    /// init that this config feeds. `nil` for single-process jobs.
    public let coordinatorAddress: String?

    public init(
        processIndex: Int,
        numProcesses: Int,
        localDeviceCount: Int,
        coordinatorAddress: String? = nil
    ) {
        precondition(numProcesses > 0, "numProcesses must be positive")
        precondition(processIndex >= 0 && processIndex < numProcesses,
                     "processIndex \(processIndex) out of range [0, \(numProcesses))")
        precondition(localDeviceCount > 0, "localDeviceCount must be positive")
        self.processIndex = processIndex
        self.numProcesses = numProcesses
        self.localDeviceCount = localDeviceCount
        self.coordinatorAddress = coordinatorAddress
    }

    /// Total devices across the whole job.
    public var globalDeviceCount: Int { numProcesses * localDeviceCount }

    /// By convention process 0 hosts the coordination service.
    public var isCoordinator: Bool { processIndex == 0 }

    /// The global device index for one of this process's local devices.
    public func globalDeviceIndex(localDevice: Int) -> Int {
        precondition(localDevice >= 0 && localDevice < localDeviceCount,
                     "localDevice \(localDevice) out of range [0, \(localDeviceCount))")
        return processIndex * localDeviceCount + localDevice
    }

    /// The global device indices owned by this process (contiguous block).
    public var localGlobalIndices: [Int] {
        (0..<localDeviceCount).map { globalDeviceIndex(localDevice: $0) }
    }
}

extension DistributedSampler {
    /// A sampler that shards a dataset across *every device of a multi-host job*.
    /// Each (process, local device) pair maps to a distinct global rank, so with a
    /// shared seed the shards are globally disjoint and cover the dataset — the
    /// same guarantee as single-host, extended across the cluster.
    public static func multiHost(
        count: Int,
        config: MultiHostConfig,
        localDevice: Int,
        shuffle: Bool = true,
        seed: UInt64 = 0,
        dropLast: Bool = false
    ) -> DistributedSampler {
        DistributedSampler(
            count: count,
            numReplicas: config.globalDeviceCount,
            rank: config.globalDeviceIndex(localDevice: localDevice),
            shuffle: shuffle, seed: seed, dropLast: dropLast)
    }
}
