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
Stable cross-language identifiers and low-level contracts.

### Fission.Plan (F#)
Typed inference IR. This is where future optimization passes will transform high-level inference intent into executable plans.

### Fission.Runtime (C#)
Hot-path state ownership, sequence lifecycle, KV page metadata, snapshots, forks, migration hooks, and eventually backend execution.

### Fission.Scheduler (F#)
Pure scheduling policy. The initial scheduler is intentionally simple: priority + deadline ordering subject to token and KV budgets.

## KV ownership model

The bootstrap runtime implements metadata-level shared KV page leases. Forking a page table increments page lease references instead of duplicating pages. Snapshots also retain leases. Actual GPU page allocation and copy-on-write writes are the next layer to implement.

## Near-term roadmap

1. Add unit/property tests for sequence transitions, KV lease ownership, snapshots, and forks.
2. Introduce `IInferenceBackend` and a deterministic fake backend.
3. Build a single-device executor loop driven by `InferencePlan`.
4. Add bounded admission queues and cancellation.
5. Integrate an ONNX Runtime backend for first-token end-to-end execution.
6. Replace metadata-only KV pages with device-backed page pools.
7. Add replayable scheduler traces and policy simulation.
