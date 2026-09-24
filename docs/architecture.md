# Fission architecture

Fission is being built as an inference runtime rather than an API wrapper.

## Design thesis

Fission treats LLM inference as an operating-system-like workload:

- a **sequence** behaves like a process;
- the **typed inference plan** is the executable plan/IR;
- **KV pages** are virtualized state;
- **snapshots and forks** are first-class operations;
- the **scheduler** allocates token and KV budgets under latency/priority constraints;
- GPU/CPU/native backends are devices behind the runtime boundary.

## Project boundaries

### Fission.Abstractions (C#)
Stable cross-language identifiers plus the device-facing `IInferenceBackend` contract.

### Fission.Plan (F#)
Typed inference IR. This is where future optimization passes will transform high-level inference intent into executable plans.

### Fission.Runtime (C#)
Hot-path state ownership, sequence lifecycle, KV page metadata, snapshots, forks, migration hooks, continuous micro-batching, and backend execution.

### Fission.Scheduler (F#)
Pure scheduling policy. The initial scheduler is intentionally simple: priority + deadline ordering subject to token and KV budgets.

## Device ownership

A `ContinuousBatchExecutor` owns one `IInferenceBackend`. Request producers never invoke the backend directly; they submit prefill/decode work through a bounded channel. A single consumer drains available work into micro-batches, preventing arbitrary request tasks from racing the device.

`DeterministicBackend` is the first backend implementation. It intentionally performs no model inference and exists to make runtime concurrency and lifecycle semantics testable before ONNX/CUDA behavior is introduced.

## KV ownership model

The bootstrap runtime implements metadata-level shared KV page leases. Forking a page table increments page lease references instead of duplicating pages. Snapshots also retain leases. Actual GPU page allocation and copy-on-write writes are the next layer to implement.

## Near-term roadmap

1. Add unit/property tests for sequence transitions, KV lease ownership, snapshots, and forks.
2. Connect `InferencePlan` compilation to executor submissions.
3. Add token-budget-aware continuous batching and cancellation propagation.
4. Integrate an ONNX Runtime backend for first-token end-to-end execution.
5. Replace metadata-only KV pages with device-backed page pools.
6. Add replayable scheduler traces and policy simulation.
