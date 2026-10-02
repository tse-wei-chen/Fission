# Serving benchmark

Fission's serving benchmark tooling is deliberately external to the inference runtime. It can target Fission, vLLM, SGLang, or another server that implements compatible streaming completion/chat-completion endpoints.

The toolchain has three layers:

1. `Fission.Serving.LoadGen` performs one measured run and writes one JSON report.
2. `eng/run-serving-suite.ps1` expands a workload manifest into repeated runs and a deterministic result-directory layout.
3. `Fission.Serving.Reporter` aggregates those JSON reports into Markdown/CSV comparison tables.

None of these projects reference Fission runtime assemblies.

## One measured run

Start the current Fission container:

```bash
docker compose up --build -d
```

Then run the load generator:

```bash
dotnet run --project benchmarks/Fission.Serving.LoadGen/Fission.Serving.LoadGen.csproj -c Release -- \
  --url http://127.0.0.1:8000 \
  --model benchmark \
  --endpoint completions \
  --requests 100 \
  --concurrency 16 \
  --max-tokens 64 \
  --warmup 4 \
  --output artifacts/serving/fission/manual/c16-r1.json
```

For chat serving use `--endpoint chat`.

Use `--prompt-file <path>` when the same prompt must be reused across different engines. If authentication is required, prefer the `OPENAI_API_KEY` environment variable rather than putting a key in shell history.

## Workload suite

The default manifest is `benchmarks/serving/workloads.json`. It currently defines:

- short interactive,
- chat,
- RAG-like,
- decode-heavy.

Each workload fixes endpoint type, prompt file, request count, output-token request, warmup, timeout, and a concurrency matrix.

Run one repetition against Fission:

```powershell
pwsh ./eng/run-serving-suite.ps1 `
  -Label fission `
  -BaseUrl http://127.0.0.1:8000 `
  -Model benchmark `
  -Repetitions 1
```

Run the same manifest against another OpenAI-compatible engine by changing only `Label`, `BaseUrl`, and the model identifier required by that endpoint.

For more stable measurements, use repeated runs:

```powershell
pwsh ./eng/run-serving-suite.ps1 `
  -Label fission `
  -BaseUrl http://127.0.0.1:8000 `
  -Model my-model `
  -Repetitions 3
```

Results are written as:

```text
artifacts/serving/
  fission/
    short-interactive/
      c1-r1.json
      c1-r2.json
      c8-r1.json
      ...
  vllm/
    short-interactive/
      ...
```

The engine label is intentionally separate from the API model name so multiple servers can expose the same model identifier.

## NVIDIA hardware gate

Once the one-shot NVIDIA real-model smoke passes, use the dedicated hardware
runner rather than manually composing environment variables and benchmark
commands:

```powershell
pwsh ./eng/run-nvidia-serving-benchmark.ps1 `
  -ModelPath <decoder-with-past.onnx> `
  -TokenizerPath <tokenizer.json> `
  -ModelId <model> `
  -NumHiddenLayers <layers> `
  -NumKvHeads <kv-heads> `
  -HeadDim <head-dim> `
  -VocabularySize <vocab>
```

The default manifest is `benchmarks/serving/workloads.gpu-smoke.json`. It is a
small functional/performance gate, not a final throughput characterization. The
runner:

1. inventories the selected NVIDIA GPU and driver;
2. records the repository commit, .NET SDK, ONNX Runtime GPU package, model
   geometry, and benchmark settings;
3. starts `Fission.Server` with ONNX Runtime CUDA and the production tokenizer;
4. keeps the startup inference probe enabled before HTTP serving;
5. waits for `/healthz`;
6. runs the existing serving suite;
7. generates Markdown and CSV reports;
8. preserves server stdout/stderr beside the results.

Each invocation writes to a timestamped directory under
`artifacts/serving-gpu/`, preventing older measurements from being silently
mixed into a new report.

After this gate is stable, use
`-Manifest benchmarks/serving/workloads.json -Repetitions 3` for the larger
concurrency matrix.

### A/B page-locked decode logits

The CUDA binding can optionally use reusable CUDA page-locked host memory for
**decode logits only**. The model still produces the same FP32 logits and CPU
greedy sampling is unchanged; the experiment isolates the host destination used
for the CUDA-to-host logits transfer.

Run a baseline and an experimental pass with identical hardware, manifest and
repetition count. Use different labels:

```powershell
# Baseline
pwsh ./eng/run-nvidia-serving-benchmark.ps1 ... `
  -Label fission-cuda

# Experimental
pwsh ./eng/run-nvidia-serving-benchmark.ps1 ... `
  -Label fission-cuda-pinned `
  -PageLockedDecodeLogits
```

Compare TPOT first, especially at concurrency 4 and 8, then output-token
throughput and TTFT. Treat a single run as directional only; use at least three
repetitions before changing the production default.

## Aggregate reports

Create Markdown and CSV summaries:

```bash
dotnet run --project benchmarks/Fission.Serving.Reporter/Fission.Serving.Reporter.csproj -c Release -- \
  --input artifacts/serving \
  --markdown artifacts/serving/report.md \
  --csv artifacts/serving/report.csv
```

To add matched percentage deltas against a baseline engine:

```bash
dotnet run --project benchmarks/Fission.Serving.Reporter/Fission.Serving.Reporter.csproj -c Release -- \
  --input artifacts/serving \
  --baseline fission \
  --markdown artifacts/serving/report.md \
  --csv artifacts/serving/report.csv
```

Baseline matching requires the same workload, endpoint, model identifier, concurrency, and requested output-token count. The reporter does not invent a comparison when those dimensions differ.

For repeated runs, throughput means and population standard deviation are reported. Percentile columns are arithmetic means of the per-run percentile values because the load generator currently stores summary statistics rather than raw per-request samples. They must not be interpreted as a pooled percentile.

## Metrics

The load generator reports:

- **TTFT**: request start until the first non-empty streamed content arrives.
- **TPOT**: `(E2E - TTFT) / (completion_tokens - 1)` for requests with at least two output tokens.
- **E2E**: request start until the SSE `data: [DONE]` terminator.
- **request throughput**: successful requests divided by measured wall-clock time.
- **output throughput**: reported completion tokens divided by measured wall-clock time.
- p50, p95, p99, and mean latency summaries.

The load generator sends `stream_options.include_usage=true`. When a server returns `usage.completion_tokens`, that value is used for TPOT and token throughput. If a server omits streaming usage, the benchmark falls back to counting non-empty content chunks and marks that run as approximate.

The reporter shows how many repeated runs had exact usage accounting.

## Fair cross-engine comparisons

For Fission vs another inference server, keep all of these constant:

1. physical GPU and driver/runtime environment,
2. model and model revision,
3. quantization and precision,
4. prompt text,
5. requested output length,
6. concurrency,
7. request count and warmup count,
8. endpoint type,
9. sampling behavior when configurable,
10. benchmark manifest and repetition count.

Run engines separately on an otherwise idle machine. Record each engine version/commit beside the generated report artifacts.

Do not compare scheduler microbenchmark numbers with serving benchmark numbers. The scheduler benchmark isolates one in-process hot path; the serving benchmark includes HTTP, queueing, scheduling, backend execution, and streaming.

## Workload limitations

The checked-in prompts are stable text fixtures, not tokenizer-normalized datasets. Their token length will differ by tokenizer/model.

Once real model serving is wired into `Fission.Server`, serious cross-engine results should use tokenizer-aware datasets or prompts selected by token count for the exact model. The current deterministic server is useful for transport, orchestration, and concurrency validation, not GPU performance conclusions.

## CI scope

`benchmarks/serving/workloads.ci.json` is intentionally tiny. Container CI runs it against the deterministic Fission image, then invokes the reporter to generate Markdown and CSV.

That CI path validates:

- manifest parsing,
- runner orchestration,
- streaming load generation,
- deterministic result layout,
- JSON aggregation,
- Markdown/CSV report generation.

Hosted-runner timing is not treated as a performance threshold.


## Workflow artifacts

Container CI writes its tiny suite under `artifacts/serving-ci/`, adds the generated Markdown report to the GitHub Actions job summary, and uploads the JSON/Markdown/CSV directory as a 14-day workflow artifact.

This makes protocol/orchestration benchmark evidence inspectable after a PR run without treating hosted-runner timing as a performance threshold. Real GPU comparison artifacts should be produced on controlled hardware using the full workload manifest.


### A/B graph-side greedy sampling

After validating page-locked decode logits, the next transfer-reduction gate is
to keep the full logits tensor inside the ONNX CUDA graph and return only one
int64 token id per batch row.

Install the ONNX Python package in the model-export environment, then append the
extra output:

```powershell
python -m pip install onnx
python ./eng/add-greedy-argmax-output.py `
  --input artifacts/models/SmolLM2-135M-Instruct/onnx/model.onnx `
  --output artifacts/models/SmolLM2-135M-Instruct/onnx/model.fission-greedy.onnx `
  --sampled-output fission_sampled_token_ids
```

The rewrite preserves the original `logits` graph output for compatibility and
adds `Gather` over the last sequence position followed by `ArgMax` over the
vocabulary. Fission does not fetch `logits` when the sampled-token output is
configured.

Benchmark the rewritten graph with the same hardware and workload:

```powershell
pwsh ./eng/run-nvidia-serving-benchmark.ps1 `
  -ModelPath artifacts/models/SmolLM2-135M-Instruct/onnx/model.fission-greedy.onnx `
  -TokenizerPath artifacts/models/SmolLM2-135M-Instruct/tokenizer.json `
  -ModelId SmolLM2-135M-Instruct `
  -NumHiddenLayers 30 `
  -NumKvHeads 3 `
  -HeadDim 64 `
  -VocabularySize 49152 `
  -EosTokenIds 2 `
  -SampledTokenIdsOutput fission_sampled_token_ids `
  -Label fission-cuda-graph-greedy `
  -Repetitions 3
```

Do not add `-PageLockedDecodeLogits` for this comparison: graph-side sampling
bypasses the full host logits output entirely, so the pinned-logits path is not
used.


### ONNX Runtime profiling gate

Use ORT profiling after an A/B result identifies a bottleneck that cannot be
explained by request scheduling alone. Profiling changes timing, so do not use
the profiled run itself as the performance comparison.

For the current validated pinned-logits path:

```powershell
pwsh ./eng/run-nvidia-serving-benchmark.ps1 `
  -ModelPath artifacts/models/SmolLM2-135M-Instruct/onnx/model.onnx `
  -TokenizerPath artifacts/models/SmolLM2-135M-Instruct/tokenizer.json `
  -ModelId SmolLM2-135M-Instruct `
  -NumHiddenLayers 30 `
  -NumKvHeads 3 `
  -HeadDim 64 `
  -VocabularySize 49152 `
  -EosTokenIds 2 `
  -PageLockedDecodeLogits `
  -OrtProfile `
  -Label fission-cuda-pinned-profile `
  -Repetitions 1
```

The runner uses a token-gated graceful shutdown so ORT can flush its profile
during session disposal. The timestamped run directory contains:

- the raw `ort-profile*.json` Chrome trace;
- `ort-profile-summary.md`;
- `ort-profile-summary.json`;
- the normal serving report and environment metadata.

The summary aggregates ORT `Node` trace event durations by execution provider
and operator and reports Memcpy execution-event count and summed duration.
Summed trace durations are diagnostic event time, not wall-clock request
latency. Use them to answer whether the CUDA/CPU partition and Memcpy nodes are
actually expensive before changing graph placement or provider options.
