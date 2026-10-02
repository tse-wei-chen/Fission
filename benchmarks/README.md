# Benchmarks

Fission keeps benchmark tools separate from runtime code.

- `Fission.Scheduler.Benchmarks`: in-process scheduler hot-path timing and allocation probe.
- `Fission.Serving.LoadGen`: external OpenAI-compatible streaming load generator for end-to-end serving measurements and cross-engine comparisons.

See [scheduler benchmark documentation](../docs/scheduler-benchmark.md) and [serving benchmark documentation](../docs/serving-benchmark.md).
