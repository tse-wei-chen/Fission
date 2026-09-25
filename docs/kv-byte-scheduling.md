# Physical KV byte scheduling

Fission schedules KV capacity along three related dimensions:

- metadata/page capacity from `KvPagePool`;
- steady-state retained physical KV bytes from an optional `IInferenceKvMemoryProfile`;
- temporary immutable successor-frontier KV bytes required while the prior state is still live.

The page budget describes the runtime's logical/paged ownership model. The byte budget describes model-specific physical KV payload. Keeping these dimensions explicit prevents models with different KV geometry or element types from being treated as if they had the same memory cost.

## Retained byte contract

`SchedulingCandidate.KvBytesPerToken` is the retained KV byte cost of one additional token for that sequence's model. A value of zero means no physical-byte profile is available, so byte admission is disabled for that candidate while token/page budgets remain active.

`SchedulingBudget.AvailableKvBytes` is the physical KV byte slack at the beginning of the scheduling cycle. `InferenceEngine` derives it from:

```text
MaxKvBytes - sum(sequence.Position * KvBytesPerToken)
```

Selected work records `KvByteGrant`:

```text
KvByteGrant = TokenGrant * KvBytesPerToken
```

This is the amount by which retained memory grows after the prior immutable state is released and the new state becomes the live frontier.

## Immutable successor-frontier peak

The current decoder state model is immutable. During a model step the prior state is still retained while the backend allocates the complete successor frontier. For a dense modeled sequence, Fission therefore also reserves:

```text
TransientKvByteGrant =
    (Position + TokenGrant) * KvBytesPerToken
```

This reservation is accounted separately from retained growth, but it competes against the same `AvailableKvBytes` slack. For multiple selected sequences, successor-frontier reservations accumulate independently from retained grants.

Example with `128 bytes/token` and `512 bytes` of current physical slack:

```text
position 3 + decode 1
retained grant  = 1 * 128 = 128
successor peak  = 4 * 128 = 512   -> admissible

position 4 + decode 1
retained grant  = 1 * 128 = 128
successor peak  = 5 * 128 = 640   -> defer
```

The second case would be unsafe even though the final retained state would fit after the old state is released. It is deferred with `TransientKvByteBudget`.

Compiled `ScheduledWorkItem` values expose both `KvByteGrant` and `TransientKvByteGrant`; `ScheduledBatch` exposes `ConsumedKvBytes` and `ConsumedTransientKvBytes` for diagnostics/replay.

## Engine feedback

`InferenceEngineOptions.MaxKvBytes` optionally configures a total physical KV byte budget. The engine receives an `IInferenceKvMemoryProfile` explicitly and recomputes retained bytes from every live runtime sequence before each scheduling cycle.

The remaining byte slack is passed to the F# scheduler. The scheduler then applies both constraints to that same starting slack:

1. final retained growth must fit;
2. complete immutable successor frontiers selected for the cycle must fit while their prior states are still live.

Explicit profile injection is intentional. The engine does not reach through runtime internals to discover a backend; a future multi-device/model router can provide a routing-aware profile without changing the scheduling kernel.

## Decoder geometry bridge

`DecoderOrtGeometryKvMemoryProfile` adapts a concrete `DecoderOrtGeometry` into the generic retained-byte profile.

For the legacy dense KV layout:

```text
[batch, kv_heads, sequence, head_dim]
```

retained bytes per sequence are:

```text
layers * 2(key+value) * kv_heads * sequence_length * head_dim * element_size
```

so bytes per token are the same expression with `sequence_length = 1`.

This naturally distinguishes GQA/MQA head counts and Float/Float16/BFloat16 KV storage without exposing ONNX Runtime types to the engine.

## Important boundary: this is not a complete device peak estimator

The successor-frontier constraint closes the largest known gap in the current dense immutable KV design, but it still does **not** include:

- model weights;
- logits buffers;
- attention/kernel workspace;
- ONNX Runtime allocator overhead and fragmentation;
- provider-specific temporary tensors;
- CUDA graph/workspace reservations.

Those resources need a provider-aware workspace/transient profile. Paged/COW KV backends should also eventually provide a more precise successor cost instead of the current conservative dense-frontier rule.

The retained and successor counters are therefore safety-oriented scheduling signals for the current physical state model, not a claim that `MaxKvBytes` represents total GPU memory usage.

## Executable invariants

Scheduler specs prove both byte constraints independently:

- 512 available bytes with a 128-byte/token model shrink an initial prefill to four tokens;
- a decode at position three is admitted because its four-token successor is exactly 512 bytes;
- a decode at position four is deferred with `TransientKvByteBudget` because its five-token successor needs 640 bytes even though its retained delta is only 128 bytes.

Existing engine specs continue to prove the retained feedback loop: executed prompt chunks advance runtime position, retained bytes reduce the next cycle's physical slack, and cancellation releases runtime/page ownership cleanly.
