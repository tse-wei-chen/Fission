# Serving benchmark

`benchmarks/Fission.Serving.LoadGen` is a dependency-free .NET load generator for OpenAI-compatible streaming endpoints.

It is intentionally outside the Fission runtime and does not reference any Fission project. The same executable can therefore target Fission, vLLM, SGLang, or another server that implements the compatible completion or chat-completion streaming shape.

## Run against Fission

Start the current container:

```bash
docker compose up --build -d
```

Then run a measured workload:

```bash
dotnet run --project benchmarks/Fission.Serving.LoadGen/Fission.Serving.LoadGen.csproj -c Release -- \
  --url http://127.0.0.1:8000 \
  --model benchmark \
  --endpoint completions \
  --requests 100 \
  --concurrency 16 \
  --max-tokens 64 \
  --warmup 4 \
  --output artifacts/serving/fission.json
```

For chat serving use `--endpoint chat`.

Use `--prompt-file <path>` when the same prompt must be reused across different engines. If authentication is required, prefer the `OPENAI_API_KEY` environment variable rather than putting a key in shell history.

## Metrics

The benchmark reports:

- **TTFT**: request start until the first non-empty streamed content arrives.
- **TPOT**: `(E2E - TTFT) / (completion_tokens - 1)` for requests with at least two output tokens.
- **E2E**: request start until the SSE `data: [DONE]` terminator.
- **request throughput**: successful requests divided by measured wall-clock time.
- **output throughput**: reported completion tokens divided by measured wall-clock time.
- p50, p95, p99, and mean latency summaries.

The load generator sends `stream_options.include_usage=true`. When a server returns `usage.completion_tokens`, that value is used for TPOT and token throughput. If a server omits streaming usage, the benchmark falls back to counting non-empty content chunks and clearly marks the report as approximate.

The JSON report records whether every successful request supplied usage data.

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
9. sampling behavior when configurable.

Run each engine separately on an otherwise idle machine. Save JSON output from every run and record the engine version/commit beside it.

Do not compare scheduler microbenchmark numbers with serving benchmark numbers. The scheduler benchmark isolates one in-process hot path; the serving benchmark includes HTTP, queueing, scheduling, backend execution, and streaming.

## Useful workload shapes

A practical first matrix is:

| Workload | Prompt | Output | Concurrency |
| --- | ---: | ---: | ---: |
| short interactive | small | 64 | 1, 8, 32 |
| chat | medium | 256 | 8, 32, 128 |
| RAG-like | long | 256 | 8, 32 |
| decode-heavy | small | 1024 | 8, 32 |
| saturation | fixed | fixed | increase until throughput stops improving |

Prompt length should be controlled with real tokenizer-aware datasets when model serving is wired into `Fission.Server`. The current deterministic server is useful for transport/concurrency validation, not GPU performance conclusions.

## CI scope

Container CI runs a tiny load-generator workload after the image health and protocol smoke tests. That CI run only proves that the benchmark client and server streaming protocol remain compatible; hosted-runner timing is not treated as a performance threshold.
