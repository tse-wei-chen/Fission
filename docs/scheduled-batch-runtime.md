# Scheduled batch runtime

Fission keeps scheduling policy and runtime execution separated by a small C#-friendly contract.

Flow:

1. `Fission.Scheduler` produces a pure `SchedulingDecision`.
2. `ScheduleCompiler` converts selected work into `ScheduledBatch`.
3. `ScheduledBatchExecutor` validates the entire batch before runtime side effects.
4. Each admitted work item becomes a one-step `CompiledExecutionPlan`.
5. Those plans are launched concurrently and converge on the single-device `ContinuousBatchExecutor`.

The bridge validates duplicate sequences, token grants, decode quantum size, prefill bindings, and aggregate token/KV accounting before execution begins.

`KvPageGrant` is currently an accounting contract. Physical device-backed KV page reservation will be wired to it in a later milestone.
