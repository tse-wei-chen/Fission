# Decoder-only ONNX execution adapter

`DecoderOnlyOnnxExecutionAdapter` is the generic stateful ONNX execution layer for decoder-style models.

It deliberately does not know Qwen, Llama, or any other export's concrete tensor geometry. Instead it composes a model-specific `IDecoderOrtModelBinding` with Fission's generic transactional decoder state.

## Responsibility split

```text
OnnxRuntimeBackend
  owns InferenceSession
  validates live graph contract
        |
        v
DecoderOnlyOnnxExecutionAdapter
  owns sequence/snapshot transaction protocol
  owns DecoderStateStore<DecoderOrtState>
  validates state-position / immutability invariants
        |
        v
IDecoderOrtModelBinding
  owns tensor packing
  owns output allocation
  executes CallerOwnedOrtRun / IO binding
  reads logits and selects next token
  returns a new immutable DecoderOrtState
```

The adapter never hard-codes model tensor names or KV shapes.

## Session contract ownership

A decoder binding exposes an `OnnxSessionContract`.

`DecoderOnlyOnnxExecutionAdapter` exposes that contract through `IOnnxRuntimeSessionContractProvider`. When `OnnxRuntimeBackendOptions.SessionContract` is not explicitly supplied, the backend validates the adapter-provided contract against the live `InferenceSession` before `InitializeAsync` reaches the binding.

This preserves fail-fast behavior without requiring callers to configure the same model manifest twice.

An explicit backend `SessionContract` remains an override for specialized hosting scenarios.

## Model-step ownership

`IDecoderOrtModelBinding.ExecutePrefill` and `ExecuteDecode` return `DecoderOrtStepResult`:

- `TokenId` is the next token selected by the binding's logits/sampling policy;
- `State` is a newly owned immutable `DecoderOrtState`;
- `IsFinished` reports model/binding termination when applicable.

On successful return, ownership of `State` transfers to the generic adapter.

A decode binding receives the prior `DecoderOrtState` as immutable input. It must not modify or dispose that state. It must create a new state version for the new position.

## Batch atomicity

The adapter separates each inference batch into two phases.

### Prepare

For every item in the batch:

1. validate sequence identity and current state;
2. execute the model binding;
3. validate the returned token/state;
4. retain the new state in an unpublished pending set.

If any model step fails, every new state produced by that batch is disposed and the `DecoderStateStore` is unchanged.

### Commit

Only after every model step succeeds does the adapter publish the new states:

- prefill adds new sequence owners;
- decode replaces the corresponding sequence owner with its new immutable version.

`ContinuousBatchExecutor` serializes backend calls, so the sequence set cannot change between prepare and commit.

## Position invariant

`DecoderOrtState.Position` is the next decoder position represented by that physical state version.

The generic adapter enforces:

- prefill result position = prompt token count;
- decode prior-state position = `DecodeItem.Position`;
- decode result position = `DecodeItem.Position + 1`.

A binding that returns the same state object from decode is rejected because decoder state versions are immutable.

## Snapshot / fork / restore

Control operations map directly into `DecoderStateStore<DecoderOrtState>`:

```text
Snapshot -> acquire current state reference
Fork     -> acquire parent state reference for each branch
Restore  -> repoint sequence to snapshot state; release divergent state
Release  -> release owner; final owner disposes physical OrtValues
```

No KV deep copy is required for snapshot or fork.

Branch divergence occurs only when a branch executes decode and the adapter installs the binding's newly returned state version for that branch.

## Executable protocol proof

`Fission.OnnxRuntime.Decoder.Specs` uses the real `mul_1.onnx` graph as a toy binding. The graph is not presented as a language model; it exists to execute the full generic protocol with real ONNX Runtime values.

The spec verifies:

1. an adapter-provided bad graph contract fails before binding initialization;
2. prefill creates one immutable physical state;
3. snapshot and fork share that state;
4. branch decode creates a divergent state;
5. restore disposes the unshared divergent state and repoints the branch to the snapshot;
6. after snapshot and parent release, the restored branch keeps the shared state alive;
7. resumed decode replaces the branch state and disposes the old shared version when its final owner disappears;
8. final sequence release disposes the latest physical state.

The binding uses `CallerOwnedOrtRun`, so the physical key/value outputs have explicit ownership and can safely become `DecoderOrtState` payloads.

## Next concrete model layer

The generic transaction protocol is now in place. A real causal-LM binding should provide model/export geometry such as:

- `input_ids`, attention-mask and position/cache-position packing;
- past/present key/value tensor layout;
- KV head count and head dimension;
- float16/bfloat16/float32 element type;
- prefill versus one-token decode shapes;
- logits shape and sampling policy;
- optional EOS/stop handling;
- CPU versus CUDA output allocation and, later, IO binding/pooling.

The first production binding should target one known export family and manifest. Other model families should add bindings/manifests rather than weakening the generic adapter into shape guessing.
