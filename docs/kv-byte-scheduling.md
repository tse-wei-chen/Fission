# Physical KV byte scheduling

Fission schedules KV capacity along two independent dimensions:

- metadata/page capacity from `KvPagePool`;
- steady-state physical KV bytes from an optional `IInferenceKvMemoryProfile`.

The page budget describes the runtime's logical/paged ownership model. The byte budget describes how much model-specific KV payload a token retains in the execution backend. Keeping both prevents a scheduler from treating models with different KV geometry or element types as if they had the same memory cost.

## Scheduling contract

`SchedulingCandidate.KvBytesPerToken` is the retained KV byte cost of one additional token for that sequence's model. A value of zero means no physical-byte profile is available, so byte admission is disabled for that candidate while token/page budgets remain active.

`SchedulingBudget.AvailableKvBytes` is the remaining retained-KV capacity for the current scheduling cycle.

Selected work records:

```text
TokenGrant
KvPageGrant
KvByteGrant
```

and the compiled `ScheduledBatch` records `ConsumedKvBytes` alongside token/page consumption.

For a modeled sequence:

```text
KvByteGrant = TokenGrant * KvBytesPerToken
```

The scheduler grants the minimum allowed by token capacity, writable KV pages, and writable KV bytes. If pages still have slack but physical bytes are exhausted, work is deferred with `KvByteBudget`.

## Engine feedback

`InferenceEngineOptions.MaxKvBytes` optionally configures a total retained-KV byte budget. The engine receives an `IInferenceKvMemoryProfile` explicitly and, before each scheduling cycle, computes retained bytes for its live requests from:

```text
sum(sequence.Position * profile.GetKvBytesPerToken(model))
```

That retained amount is subtracted from `MaxKvBytes` before calling the F# scheduler. This makes a partial prefill consume physical capacity in the next cycle even when the current metadata page still has unused token slots.

Explicit profile injection is intentional. The engine does not reach through runtime internals to discover a backend; a future multi-device/model router can provide a routing-aware profile without changing the scheduling kernel.

## Decoder geometry bridge

`DecoderOrtGeometryKvMemoryProfile` adapts a concrete `DecoderOrtGeometry` into the generic profile contract.

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

## Important boundary: retained bytes are not peak bytes

The current byte budget is deliberately a **steady-state retained KV** budget. It does not include:

- model weights;
- logits buffers;
- attention/workspace scratch;
- ONNX Runtime allocator overhead;
- temporary output buffers;
- the transient overlap between an immutable prior decoder state and its newly produced successor state.

The current FP32 Optimum binding allocates a complete present-KV tensor before the old immutable state is released, so its model-step peak can exceed the retained delta admitted by this scheduler budget.

That is not hidden by this API. A future provider-aware memory layer should add a distinct transient/peak or workspace budget, and GPU IO-binding/paged-KV implementations can reduce that peak without changing the retained-KV scheduling contract.

## Executable invariants

The scheduler specs prove that 512 available bytes with a 128-byte/token model shrink a larger prefill to four tokens even when token/page capacity is higher.

The engine specs prove the feedback loop end to end:

1. a 10-token prompt receives a four-token first prefill grant;
2. that work carries a 512-byte `KvByteGrant`;
3. the runtime sequence advances to position four;
4. the next cycle observes all 512 bytes as retained;
5. further prefill is deferred by `KvByteBudget` even though its metadata page still has token slack;
6. cancellation releases the runtime/page ownership cleanly.
