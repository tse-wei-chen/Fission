# Benchmarks

Fission keeps benchmark tools separate from runtime code.

- `Fission.Scheduler.Benchmarks`: in-process scheduler hot-path timing and allocation probe.
- `Fission.Serving.LoadGen`: external OpenAI-compatible streaming load generator for one measured serving run.
- `Fission.Serving.Reporter`: aggregates repeated serving JSON reports into Markdown and CSV, with optional matched-baseline deltas.
- `serving/workloads.json`: reusable workload/concurrency matrix.
- `../eng/run-serving-suite.ps1`: executes the workload matrix against any OpenAI-compatible endpoint.

See [scheduler benchmark documentation](../docs/scheduler-benchmark.md) and [serving benchmark documentation](../docs/serving-benchmark.md).
