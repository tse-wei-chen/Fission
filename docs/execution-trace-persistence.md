# Execution trace persistence

Fission execution traces can be exported to and restored from a versioned JSON Lines format through `ExecutionTraceJsonLines`.

The persistence layer is intentionally separate from `ExecutionTraceReplay`: serialization owns portability and schema validation, while replay continues to own deterministic runtime-state reconstruction. This keeps future trace storage choices independent from scheduler/replay semantics.

## Basic workflow

```csharp
var trace = new InMemoryExecutionTraceSink();

// Pass trace into ExecutionPlanExecutor and execute workload...

await ExecutionTraceJsonLines.WriteFileAsync(
    "production.trace",
    trace.Snapshot());

var recorded = await ExecutionTraceJsonLines.ReadFileAsync(
    "production.trace");

var replay = ExecutionTraceReplay.Replay(recorded);
```

The same codec also accepts caller-owned streams through `WriteAsync` and `ReadAsync`. Stream overloads leave the stream open.

## Format

The file is UTF-8 JSON Lines. Line 1 is a schema header:

```json
{"recordType":"header","schema":"fission.execution-trace","version":1}
```

Every following line is one recorded event and includes its monotonic ordinal plus the complete `ExecutionTraceEvent` payload. Version 1 preserves:

- plan id, step index, operation, and event kind;
- sequence / related-sequence / snapshot identity;
- position, KV-page count, and device placement;
- migration source/target, transaction id, transport id/kind, and byte estimate;
- planner-estimated and measured durations at tick precision;
- failure type and rollback-failure count.

Enums are stored by symbolic name instead of numeric ordinal so traces stay inspectable by humans and fail clearly when a reader encounters an event or transport kind it does not understand.

## Integrity rules

Writers and readers both require strictly increasing positive ordinals. Readers also validate the schema name and exact schema version, reject unknown event/transport kinds, reject empty event lines, and require a non-empty operation name.

These checks are deliberate: replay and policy simulation should fail on malformed or unsupported evidence rather than silently reconstructing a different execution history.

## Scope

`WriteFileAsync` creates/truncates a trace file and flushes the encoded snapshot before returning. This provides a portable durable artifact once export completes, but `InMemoryExecutionTraceSink` still retains live events in process memory until that export occurs.

For very long-running production servers, a later milestone should add a non-blocking streaming trace sink with bounded buffering, rotation, failure policy, and periodic durability without putting file I/O on the inference hot path. The versioned JSONL codec introduced here can remain the interchange format for that sink.
