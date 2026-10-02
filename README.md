# Fission

Fission is an experimental high-concurrency LLM inference runtime for .NET.

The project explores a typed, stateful inference architecture where:

- **F#** expresses inference plans, scheduling policy, and optimization passes.
- **C#** hosts the runtime, sequence lifecycle, memory/KV management, and serving hot paths.
- Native GPU backends can be integrated behind stable runtime abstractions.

The long-term direction is an inference operating system with first-class support for forkable KV state, snapshots, migration, scheduling contracts, and replayable execution plans.

> Status: early bootstrap.

## Container quick start

Build and start the current `Fission.Server` composition with Docker Compose:

```bash
docker compose up --build -d
curl --fail http://localhost:8000/healthz
```

The current server executable uses the deterministic backend/token codec, so this image is intended for deployment and serving integration validation rather than production GPU model serving.

See [`docs/container.md`](docs/container.md) for configuration, smoke tests, and the planned GPU-container boundary.

## Serving benchmark

Run the dependency-free OpenAI-compatible load generator against Fission or another compatible server:

```bash
dotnet run --project benchmarks/Fission.Serving.LoadGen/Fission.Serving.LoadGen.csproj -c Release -- \
  --url http://127.0.0.1:8000 \
  --model benchmark \
  --requests 100 \
  --concurrency 16 \
  --max-tokens 64 \
  --output artifacts/serving/result.json
```

It reports TTFT, TPOT, E2E latency, request throughput, output-token throughput, and p50/p95/p99 summaries.

See [`docs/serving-benchmark.md`](docs/serving-benchmark.md) for cross-engine comparison guidance.
