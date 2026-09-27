# Execution trace persistence

Fission execution traces can be exported to and restored from a versioned JSON Lines format through `ExecutionTraceJsonLines`.

The persistence layer is intentionally separate from `ExecutionTraceReplay`: serialization owns portability and schema validation, while replay continues to own deterministic runtime-state reconstruction. This keeps future trace storage choices independent from scheduler/replay semantics.

## Snapshot workflow

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

## Streaming workflow

Long-running servers can use `StreamingExecutionTraceSink` instead of retaining the whole trace in memory:

```csharp
await using var trace = new StreamingExecutionTraceSink(
    "traces",
    new StreamingExecutionTraceOptions
    {
        Capacity = 4096,
        MaxEventsPerSegment = 1024,
        FlushInterval = TimeSpan.FromSeconds(1),
        OverflowPolicy = ExecutionTraceBufferOverflowPolicy.DropNewest,
        BackgroundFailurePolicy = ExecutionTraceBackgroundFailurePolicy.Isolate,
        DurabilityPolicy = ExecutionTraceDurabilityPolicy.SegmentFlush
    });

// Pass trace into ExecutionPlanExecutor and execute workload...
```

`Record` never waits for file I/O. It assigns the event ordinal and attempts a bounded-channel `TryWrite` inside a short ordering critical section. JSON serialization, file creation, flushing, rotation, and optional physical disk flush happen on one background writer.

The sink writes self-contained version-1 JSONL segments. A segment is published under its final `.trace` name only after the temporary file has been completely encoded and closed. Segment names include the sink session id, segment index, and first/last ordinal, so lexical segment order is also execution order for one session.

Rotation happens when `MaxEventsPerSegment` is reached. Low-volume traffic is flushed when `FlushInterval` expires, so a server does not need to fill a segment before obtaining a portable trace artifact. `DisposeAsync` completes the channel, drains remaining events, writes the final partial segment, and surfaces any background persistence failure.

## Backpressure and failure policy

The streaming buffer is intentionally bounded. `ExecutionTraceBufferOverflowPolicy.DropNewest` keeps inference isolated from trace pressure and increments `DroppedEventCount`; `Throw` makes a full buffer visible immediately to the caller. Dropped events still consume ordinals before they are rejected, so later persisted segments expose ordinal gaps rather than pretending the trace is complete.

Background storage failures are always observable through `BackgroundFailure`, `Completion`, and `DisposeAsync`. With `ExecutionTraceBackgroundFailurePolicy.Isolate`, later `Record` calls are counted as dropped and inference can continue. `ThrowOnRecord` instead turns an already-observed background writer failure into a synchronous error on subsequent trace recording.

`ExecutionTraceDurabilityPolicy.SegmentFlush` relies on the normal close/flush boundary of each completed JSONL segment. `FlushToDisk` additionally issues `FileStream.Flush(flushToDisk: true)` on the background writer before publishing the segment. Neither policy puts disk I/O on the inference caller thread.

Operational counters are exposed as `PersistedEventCount`, `DroppedEventCount`, and `LastPersistedOrdinal`; `SegmentFiles` lists the completed segment paths for the current session.

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

Streaming segment files individually satisfy the same format and integrity rules. When replaying multiple segments, concatenate them in segment-index order before calling `ExecutionTraceReplay.Replay`. Ordinal gaps are valid syntax but indicate that the streaming sink dropped or otherwise lost evidence; production tooling should surface that distinction instead of treating the trace as complete.

## Scope

`InMemoryExecutionTraceSink` plus `WriteFileAsync` remains useful for bounded tests and explicit point-in-time exports. `StreamingExecutionTraceSink` covers long-running production capture with bounded buffering, rotation, failure isolation, and periodic durability while reusing the same versioned JSONL interchange format.

The next trace-specific layer can build session manifests/indexes and direct multi-segment replay tooling without changing the runtime event schema.
