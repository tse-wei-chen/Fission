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
