# Stateful chunked prefill

Fission may split one prompt into multiple prefill chunks before decode begins. A stateful backend must therefore treat a later prefill chunk as an append to the sequence's current immutable decoder state, not as another prompt starting from empty KV.

## Backend work item

`PrefillItem.Position` is optional. When supplied, it is the number of prompt tokens already committed before the chunk starts. When omitted, a stateful backend may infer the continuation position from the decoder state it already owns. This preserves the existing runtime call path while still allowing direct callers to request strict position validation.

A sequence with no backend state can only begin at position zero. A sequence with backend state can only continue at that state's exact position.

## Decoder binding capability

`IDecoderOrtChunkedPrefillModelBinding` is an optional extension of `IDecoderOrtModelBinding`. The generic decoder adapter requires this capability before it executes any model work when a prefill batch contains an existing sequence state.

The adapter validates the complete batch first:

- sequence ids are unique,
- every chunk contains at least one token,
- explicit positions are non-negative,
- new sequences start at zero,
- continuation positions match the owned decoder state, and
- every continuation is supported by the binding.

Only after those checks succeed does model execution begin.

## Immutable state transition

A continuation receives the current immutable `DecoderOrtState` and must return a new state version at:

`priorState.Position + item.Tokens.Length`.

The prior state is never mutated in place. After all model steps in the prefill batch validate, the adapter adds state for new sequences and replaces state for continuation sequences through `DecoderStateStore`.

Snapshots and forked branches that still reference the prior version keep that physical state alive through the existing reference-counted ownership graph.

## Prompt tokens versus generated-token frontier

A prefill continuation consumes the actual prompt tokens in `PrefillItem.Tokens`. It must not feed `priorState.NextTokenId` as though the operation were a decode step.

The new state stores the sampled token from the final prompt position as its `NextTokenId`. Decode after the final prompt chunk consumes that frontier normally.

For the concrete `OptimumLegacyFloatDecoderBinding`, continuation reuses the same scalar `ExecuteStep` path as ordinary prefill/decode: prior KV is bound as past state, position ids start at the prior state position, present KV becomes a new immutable state, and logits from the final chunk position establish the next-token frontier.

## Current batching scope

This change fixes stateful correctness for chunked prompt execution. Prefill chunks are still executed through the scalar decoder binding path; decode has the separate same-position cohort batching optimization. Batched prefill is a future throughput optimization and must preserve the continuation invariants above.
