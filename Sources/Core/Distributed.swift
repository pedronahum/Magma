// Magma - High-level distributed ops
// Tensor-level ergonomics for data-parallel / SPMD execution: cross-replica
// collectives written in normal Tensor code, a distributable input factory, and
// graph extraction. A DDP gradient sync becomes `grad.crossReplicaMean(...)`, and
// the computation runs across devices via `executeGraphReplicated` on the graph
// from `.makeGraph()`.

import LazyTensor
import XLARuntime

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
