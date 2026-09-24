# ONNX decoder state store

A causal-LM ONNX adapter must keep model-owned decoder state alive between prefill and decode steps. For cached decoder exports that state usually includes past/present key-value tensors and may also include adapter-specific metadata.

`DecoderStateStore<TState>` provides the ownership graph for those physical state versions without defining their tensor layout.

## Immutable state versions

The store treats every decoder state payload as immutable after registration.

A model step should produce a new state version from the model's present-KV outputs and call `ReplaceSequence`. The sequence then owns the new version. Any snapshots or sibling branches that still reference the previous version continue to see the previous state.

This avoids in-place mutation of shared KV state and makes branch divergence explicit.

## Reference ownership

Every registered sequence and every registered snapshot owns one reference to a state version.

- `AddSequence` creates the first owning reference.
- `SnapshotSequence` acquires another reference to the sequence's current version.
- `ForkSequence` gives every branch a reference to the parent's current version.
- `ReplaceSequence` swaps only that sequence to a new version and releases its previous reference.
- `RestoreSequence` acquires the snapshot version for the sequence and releases the sequence's divergent version.
- `ReleaseSequence` and `ReleaseSnapshot` each drop one owning reference.
- `Dispose` releases all remaining owners as the backend-shutdown fallback.

The `TState.Dispose()` call occurs exactly once, when the final reference to that state version disappears.

## Ownership transfer

`AddSequence` and `ReplaceSequence` transfer ownership of the supplied state only when registration succeeds.

If a sequence does not exist for `ReplaceSequence`, or an add conflicts with an existing sequence, the caller still owns the supplied payload and is responsible for disposing or reusing it.

This rule is important for ORT outputs: a failed store mutation must not silently consume an `OrtValue` collection that the adapter still needs to clean up.

## Snapshot and fork cost

Snapshot and fork are metadata/reference-count operations in this store. They do not copy the physical payload.

The physical cost of a later divergent decode is paid when the adapter creates the next immutable present-KV state. This gives the adapter a natural path toward copy-on-write or paged physical KV without changing Fission's snapshot/fork transaction API.

## Integration with backend transactions

`DecoderStateStore<TState>` is intended to live inside an `IOnnxRuntimeExecutionAdapter` implementation.

The existing device actor invokes backend transaction hooks in order:

- snapshot -> `SnapshotSequence`;
- fork -> `ForkSequence`;
- restore -> `RestoreSequence`;
- snapshot release -> `ReleaseSnapshot`;
- sequence release -> `ReleaseSequence`.

Because the device actor serializes inference and transaction barriers, the store can represent the same state version observed by the corresponding model step.

## Next physical payload

The next adapter layer can define an owned payload such as:

```text
DecoderOrtState
  Position
  Layers[]
    Key   -> owned OrtValue/buffer
    Value -> owned OrtValue/buffer
```

That payload can be stored directly in `DecoderStateStore<DecoderOrtState>`.

The state store deliberately does not decide whether those tensors are backed by CPU memory, CUDA memory, IO binding, preallocated buffers, paged KV, or another provider-specific representation. Those choices belong to the execution adapter and provider backend.
