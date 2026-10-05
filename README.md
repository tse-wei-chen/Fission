# Fission

Fission is an experimental high-concurrency LLM inference runtime for .NET.

The project explores a typed, stateful inference architecture where:

- **F#** expresses inference plans, scheduling policy, and optimization passes.
- **C#** hosts the runtime, sequence lifecycle, memory/KV management, and serving hot paths.
- Native GPU backends can be integrated behind stable runtime abstractions.

The long-term direction is an inference operating system with first-class support for forkable KV state, snapshots, migration, scheduling contracts, and replayable execution plans.

> Status: early bootstrap.

## Container quick start

Portable/default serving uses the standard Compose file:

```bash
cp .env.example .env
docker compose up --build -d
curl --fail http://localhost:8000/healthz
```

The default container uses the deterministic backend/token codec so CI remains self-contained. The same portable image can be configured for ONNX Runtime CPU serving.

NVIDIA CUDA serving has a separate user-facing Compose file and env template:

```bash
cp .env.cuda.example .env.cuda
# Fill in model/tokenizer geometry in .env.cuda first.
docker compose --env-file .env.cuda -f compose.cuda.yaml up --build -d
curl --fail http://localhost:8000/healthz
```

Keeping the CUDA configuration separate avoids exposing NVIDIA-only runtime, profiling, and memory-pool options to portable users.

See [`docs/container.md`](docs/container.md) for model configuration, CUDA requirements, startup probes, profiling, and accelerator-specific guidance.

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

## NVIDIA real-model smoke

The server can run its full tokenizer -> scheduler/runtime -> ONNX Runtime CUDA
path as a one-shot startup probe and exit before opening an HTTP listener. The
helper script validates the NVIDIA host and supplies the production composition
settings:

```powershell
pwsh ./eng/run-nvidia-smoke.ps1 -ModelPath <decoder-with-past.onnx> -TokenizerPath <tokenizer.json> -ModelId <model> -NumHiddenLayers <n> -NumKvHeads <n> -HeadDim <n> -VocabularySize <n>
```

See [`docs/nvidia-smoke.md`](docs/nvidia-smoke.md) for the model contract,
CUDA/ONNX Runtime prerequisites, and expected success criteria.

## NVIDIA serving benchmark gate

After the one-shot CUDA smoke passes, run the controlled GPU serving gate to
start the real model server, wait for the startup inference probe, execute a
small concurrency matrix, capture environment metadata, and generate
Markdown/CSV results:

```powershell
pwsh ./eng/run-nvidia-serving-benchmark.ps1 -ModelPath <model.onnx> -TokenizerPath <tokenizer.json> -ModelId <model> -NumHiddenLayers <n> -NumKvHeads <n> -HeadDim <n> -VocabularySize <n>
```

The checked-in GPU smoke manifest is deliberately small. Pass the full
`benchmarks/serving/workloads.json` manifest only after this hardware gate is
stable.
