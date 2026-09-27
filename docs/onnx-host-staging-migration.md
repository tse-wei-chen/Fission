# ONNX Runtime host-staging migration

Fission now has a first concrete physical implementation of the cross-device migration protocol for decoder state owned by the ONNX Runtime backend.

The implementation is intentionally opt-in. `OnnxRuntimeBackend` keeps its existing non-transactional behavior, while `OnnxRuntimeMigratableBackend` requires an execution adapter that implements `IOnnxRuntimeSequenceMigrationAdapter` and advertises an actual physical-state codec.

## Data path

For `DecoderOnlyOnnxExecutionAdapter`, migration is enabled when its binding implements `IDecoderOrtHostStagingBinding`.

```text
source DecoderOrtState
        |
        | deep-copy K/V tensors
        v
managed host-staging payload
        |
        | transport-plan attestation
        v
target binding import
        |
        | create target-owned OrtValue views
        v
target DecoderOrtState
```

Prepare exports a deep copy of every K/V tensor plus the decoder causal frontier (`Position` and `NextTokenId`). Import reconstructs a distinct target `DecoderOrtState`. Commit releases source physical state only after target import succeeds. Abort removes target state and, if a destructive source commit failed, can reconstruct the source state from the staged payload.

The imported state retains a lifetime anchor to the managed staging payload until its `OrtValue` views have been disposed. This prevents the arrays backing `OrtValue.CreateTensorValueFromMemory` from becoming unreachable while ONNX Runtime still references them.

## Optimum legacy FP32 codec

`OptimumLegacyFloatHostStagingBinding` decorates `OptimumLegacyFloatDecoderBinding` and provides the first production-shaped codec. It:

- preserves the existing prefill/decode/batched execution behavior;
- estimates transfer bytes from decoder geometry;
- deep-copies FP32 key/value tensors into managed arrays on export;
- validates layer count, geometry, byte count, format id, position, and next-token frontier;
- creates distinct target `OrtValue` instances over the staged arrays on import;
- advertises a format-specific host-staging transport id so incompatible source/target bindings fail transport negotiation before export.

The transport id includes model identity and decoder geometry/format. Host staging therefore participates in the same deterministic transport planner and byte/concurrency admission control as future CUDA P2P, IPC, NIXL, or RDMA transports.

## Scope and performance

This path is a correctness-first physical transport. It proves that Fission can move real backend-owned state between independent device actors while preserving transactional placement semantics.

It is not intended to be the final high-performance GPU path. The current codec performs managed host copies and uses a static capability estimate. Future transports should replace or complement this path with pinned buffers, async device copies, CUDA P2P/IPC, NIXL/RDMA, measured topology inputs, and transfer telemetry without changing the runtime transaction protocol.

## Executable specs

`tests/Fission.OnnxRuntime.Migration.Specs` covers:

- end-to-end migration through `ExecutionPlanExecutor` and two device actors;
- source-state release only after successful target import/commit;
- target decode continuing from the migrated causal frontier;
- byte/concurrency admission release after migration;
- negotiation failure for incompatible codec ids before physical export;
- Optimum FP32 host-staging export/import round-trip and source/target memory independence.
