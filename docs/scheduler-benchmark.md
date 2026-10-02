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

These numbers are primarily for local before/after comparisons. CI also runs a short reporting sample so allocation changes are visible in build logs, but there is no timing or allocation threshold: hosted-runner timing is noisy. Allocation deltas are generally more stable than wall-clock deltas, but should still be compared on the same runtime and machine.


## Comparing scheduler changes

For a scheduler hot-path change, compare the same scenario and iteration count before and after the code change. Prefer `bytes/op` as the first signal because hosted-runner timing can vary between jobs.

When using CI for an A/B check, capture a benchmark-neutral baseline run from the current `develop` state, then run the changed branch with the same benchmark source and iteration count. Treat `ns/op` as supporting evidence rather than a merge threshold.


## CI evidence

The build workflow writes the short scheduler benchmark sample to `artifacts/benchmarks/scheduler.txt`, mirrors it into the GitHub Actions job summary, and uploads it as a 14-day workflow artifact.

The artifact is evidence, not a performance gate. Hosted-runner timing is noisy; compare allocation and timing only against runs collected under the same environment and benchmark source.
