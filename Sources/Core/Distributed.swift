// Magma - High-level distributed ops
// Tensor-level ergonomics for data-parallel / SPMD execution: cross-replica
// collectives written in normal Tensor code, a distributable input factory, and
// graph extraction. A DDP gradient sync becomes `grad.crossReplicaMean(...)`, and
// the computation runs across devices via `executeGraphReplicated` on the graph
// from `.makeGraph()`.

import LazyTensor
import XLARuntime
import _Differentiation

extension Optimizer {
    /// Data-parallel gradient sync: average each gradient across the replica
    /// groups (`crossReplicaMean`) before applying it. Drop-in for `step(_:)` in a
    /// DDP loop so replicated parameters receive an identical update and stay in
    /// sync. `groups` defaults to a single all-replica group of `numReplicas`.
    public mutating func step(syncing gradients: [Tensor<Float>], groups: [[Int]]) {
        step(gradients.map { $0.crossReplicaMean(groups: groups) })
    }
}

/// Run one data-parallel SGD step, end to end, via autodiff.
///
/// Computes `gradient(of: loss)` wrt `w` (autodiff), averages it across replicas,
/// applies `w - lr * grad`, and executes the update across `numReplicas` devices
/// — `dataDistribution` maps the per-replica data buffers captured by `loss` to
/// their per-replica values (`w` and other unlisted inputs default to
/// replicated). Returns each replica's updated parameter; all are identical (the
/// DDP invariant), and equal to a single-device full-batch step for mean losses.
public func dataParallelSGDStep(
    w: Tensor<Float>,
    lr: Float,
    numReplicas: Int,
    client: PJRTClient,
    dataDistribution: [ObjectIdentifier: ReplicaInputDistribution],
    groups: [[Int]]? = nil,
    loss: @differentiable(reverse) (Tensor<Float>) -> Tensor<Float>
) throws -> [[Float]] {
    let replicaGroups = groups ?? [Array(0..<numReplicas)]
    let grad = gradient(at: w, of: loss)                       // autodiff (lazy)
    let synced = grad.crossReplicaMean(groups: replicaGroups)  // DDP grad sync
    let lrT = Tensor<Float>.full(w.shape, lr, on: w.device)
    let wNew = w - lrT * synced                                // SGD update
    let outs = try executeGraphReplicated(
        wNew.makeGraph(), numReplicas: numReplicas,
        distribution: dataDistribution, client: client)
    return try outs.map { try $0[0].toFloatArray() }
}

extension Tensor {
    /// Cross-replica sum: every replica in a group ends with the sum of the
    /// group's per-replica values. Traces into a `stablehlo.all_reduce`; run the
    /// resulting graph data-parallel (see `makeGraph()` / `executeGraphReplicated`).
    public func crossReplicaSum(groups: [[Int]]) -> Tensor {
        collective(op: .allReduce, attributes: ["reduction": "add", "replicaGroups": groups])
    }

    /// Cross-replica mean (`all_reduce` add then scale by 1/group size) — the DDP
    /// gradient-averaging op. After it, every replica holds the mean of the
    /// per-replica inputs, so applying it as a gradient keeps replicas in sync.
    public func crossReplicaMean(groups: [[Int]]) -> Tensor {
        collective(op: .allReduceMean, attributes: ["replicaGroups": groups])
    }

    private func collective(op: OpKind, attributes: [String: Any]) -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(id: id, shape: shape, dtype: dtype, device: device)
        handle.irNode = .operation(op: op, inputs: [self.handle], attributes: attributes)
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }
}

extension Tensor {
    /// A distributable graph input backed by an on-device buffer (a `.data` node),
    /// as opposed to `Tensor(_:shape:)` which creates an inlined constant. Use for
    /// tensors that should be replicated or sharded per replica by
    /// `executeGraphReplicated` — keyed by the buffer's identity.
    public static func input(from buffer: PJRTBuffer) -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(id: id, shape: buffer.shape, dtype: Scalar.dtype, device: .default)
        handle.irNode = .data(buffer)
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Build the `IRGraph` that produces this tensor (with this tensor as the sole
    /// output), ready for compilation/execution — e.g. `executeGraphReplicated`.
    public func makeGraph() -> IRGraph {
        let graph = IRGraph()
        graph.addOutput(self.handle)
        graph.buildTopologicalOrder()
        return graph
    }
}
