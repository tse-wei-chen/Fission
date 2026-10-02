# Scheduler microbenchmark

`benchmarks/Fission.Scheduler.Benchmarks` is a lightweight allocation and throughput probe for the public scheduling-kernel boundary.

It intentionally has no external benchmark package dependency. The goal is to make scheduler hot-path work measurable with the same .NET SDK already used by the repository, while keeping CI deterministic.

## Run

```bash
dotnet run --project benchmarks/Fission.Scheduler.Benchmarks/Fission.Scheduler.Benchmarks.csproj -c Release -- 20000
```

The optional argument is the measured iteration count. The benchmark performs a bounded warmup before each scenario.

Reported columns:

- `ns/op`: elapsed wall-clock nanoseconds per `ISchedulingKernel.Schedule` call.
- `bytes/op`: managed bytes allocated on the current thread per scheduling call, measured with `GC.GetAllocatedBytesForCurrentThread`.
- `checksum`: a result-consumption guard so benchmark calls remain observably used.

Current scenarios cover a single decode, a 32-sequence mixed single-device batch, and a 128-sequence mixed four-device batch.

These numbers are for local before/after comparisons. They are not a CI performance gate because hosted-runner timing is noisy. Allocation deltas are generally more stable than wall-clock deltas, but should still be compared on the same runtime and machine.
