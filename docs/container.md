# Container development

Fission ships a reproducible container entry point for `Fission.Server`.

The default container still uses `DeterministicBackend` and
`DeterministicTextTokenCodec` so CI remains self-contained. The server
composition root can now select ONNX Runtime CPU or CUDA plus a Hugging Face
tokenizer, but the default `Dockerfile` intentionally does not install NVIDIA
CUDA/cuDNN user-space libraries. Real-model CUDA validation is therefore run on
an NVIDIA host before a dedicated GPU image is treated as supported.

## Build the image

```bash
docker build -t fission:dev .
```

The build uses the .NET 10 SDK and publishes `Fission.Server` into the .NET ASP.NET runtime image. The final image runs as the non-root user provided by the official .NET image and listens on port `8000`.

## Run with Docker

```bash
docker run --rm \
  --name fission \
  -p 8000:8000 \
  fission:dev
```

Check readiness:

```bash
curl --fail http://localhost:8000/healthz
```

Exercise the OpenAI-compatible completion endpoint:

```bash
curl --fail \
  -H 'Content-Type: application/json' \
  -d '{"model":"container-smoke","prompt":"hello","max_tokens":2,"stream":false}' \
  http://localhost:8000/v1/completions
```

## Run with Docker Compose

Copy the configuration template if local overrides are needed:

```bash
cp .env.example .env
```

Then build and start the service:

```bash
docker compose up --build -d
```

Useful lifecycle commands:

```bash
docker compose logs -f server
docker compose ps
docker compose down
```

## Configuration

`compose.yaml` maps environment variables to the ASP.NET Core configuration keys already consumed by `Fission.Server`.

| `.env` variable | Server configuration | Default |
| --- | --- | ---: |
| `FISSION_PORT` | host port mapping | `8000` |
| `FISSION_DEVICE` | `Fission:Device` | `cpu:0` |
| `FISSION_KV_PAGES` | `Fission:KvPages` | `16384` |
| `FISSION_TOKENS_PER_KV_PAGE` | `Fission:TokensPerKvPage` | `16` |
| `FISSION_MAX_BATCH_TOKENS` | `Fission:MaxBatchTokens` | `2048` |
| `FISSION_MAX_BATCH_SEQUENCES` | `Fission:MaxBatchSequences` | `128` |
| `FISSION_MAX_PREFILL_CHUNK_TOKENS` | `Fission:MaxPrefillChunkTokens` | `512` |
| `FISSION_ADMISSION_CAPACITY` | `Fission:AdmissionCapacity` | `1024` |

ASP.NET Core converts double underscores in environment names such as `Fission__KvPages` into configuration sections such as `Fission:KvPages`.

## CI validation

`.github/workflows/container.yml` validates the Compose model, builds the image, starts it, waits for `/healthz`, sends an OpenAI-compatible request, and runs a small concurrent streaming workload through the serving load generator.

`.github/workflows/image.yml` separately validates the publication build path. Pull requests build without pushing. Explicit release tags and manual dispatches can publish to GHCR.

See [releases and container publication](releases.md) for tag rules, `master` promotion, and GHCR behavior.

## NVIDIA host smoke before a GPU image

Before packaging a CUDA image, validate the exact ONNX model, tokenizer, CUDA
runtime, and driver on the target NVIDIA host:

```powershell
pwsh ./eng/run-nvidia-smoke.ps1 -ModelPath <decoder-with-past.onnx> -TokenizerPath <tokenizer.json> -ModelId <model> -NumHiddenLayers <n> -NumKvHeads <n> -HeadDim <n> -VocabularySize <n>
```

The script uses the startup probe's one-shot mode, so success means the same
production tokenizer/backend/runtime path completed inference and the process
exited cleanly without starting Kestrel. See [NVIDIA real-model smoke](nvidia-smoke.md).

## Planned GPU image

Do not simply add CUDA libraries to the default image and call it GPU serving.
Once the NVIDIA host smoke is green for a known model/runtime matrix, add a
separate image such as:

```text
Dockerfile.cuda
  -> CUDA runtime
  -> ONNX Runtime GPU/native dependencies
  -> Fission.Server
  -> mounted/read-only model directory
```

That keeps container packaging aligned with actual runtime capability.
