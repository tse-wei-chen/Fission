# Container development

Fission has two container lanes:

- `Dockerfile`: portable/default image for deterministic serving and ONNX Runtime CPU composition.
- `Dockerfile.cuda`: NVIDIA CUDA image for the production ONNX Runtime CUDA composition that is now wired through `Fission.Server`.

Model files are **not** baked into either image. `models/` is excluded from the Docker build context and Compose mounts the selected model directory read-only at `/models`.

## Portable/default image

Build:

```bash
docker build -t fission:dev .
```

Run the self-contained deterministic default:

```bash
docker run --rm -p 8000:8000 fission:dev
```

Or with Compose:

```bash
cp .env.example .env
docker compose up --build -d server
```

The portable service defaults to:

```text
Backend=deterministic
ExecutionProvider=cpu
Tokenizer=deterministic
Device=cpu:0
```

Those values can be changed through `.env`. The same image can host ONNX Runtime CPU serving when a compatible model/tokenizer is mounted and the required model geometry is configured.

## NVIDIA CUDA image

`Dockerfile.cuda` uses an NVIDIA CUDA + cuDNN runtime base and copies the .NET 10 ASP.NET runtime into that Ubuntu 24.04 environment.

The repository currently references `Microsoft.ML.OnnxRuntime.Gpu 1.30.0`. The default NuGet GPU package for ONNX Runtime 1.30 uses CUDA 13.x and cuDNN 9.x. The checked-in CUDA image therefore defaults to:

```text
nvidia/cuda:13.4.2-cudnn-runtime-ubuntu24.04
```

The CUDA runtime image is a build argument and may be overridden for a validated environment:

```bash
docker build \
  -f Dockerfile.cuda \
  --build-arg CUDA_RUNTIME_IMAGE=nvidia/cuda:13.4.2-cudnn-runtime-ubuntu24.04 \
  -t fission:cuda-dev .
```

The host still requires a compatible NVIDIA driver and NVIDIA Container Toolkit.

### Compose CUDA profile

Put the decoder-with-past ONNX graph and matching `tokenizer.json` under the directory configured by `FISSION_MODEL_DIR` (default `./models`), then fill in the model geometry in `.env`:

```dotenv
FISSION_MODEL_DIR=./models
FISSION_MODEL_FILE=decoder_with_past_model.onnx
FISSION_TOKENIZER_FILE=tokenizer.json
FISSION_MODEL_ID=my-model
FISSION_NUM_HIDDEN_LAYERS=<layers>
FISSION_NUM_KV_HEADS=<kv-heads>
FISSION_HEAD_DIM=<head-dim>
FISSION_VOCABULARY_SIZE=<vocabulary-size>
FISSION_EOS_TOKEN_IDS=<comma-separated-eos-ids>
```

Start only the CUDA service:

```bash
docker compose --profile cuda up --build -d server-cuda
```

It listens on host port `8001` by default so it can coexist with the portable service. Override with `FISSION_CUDA_PORT`.

Check health:

```bash
curl --fail http://localhost:8001/healthz
```

The service requests GPU access through Compose `gpus: all`. `FISSION_CUDA_DEVICE_ID` selects the CUDA ordinal visible inside the container; `FISSION_CUDA_DEVICE` is the Fission runtime identity and defaults to `cuda:0`.

## CUDA optimization switches

The Compose CUDA profile exposes the CUDA serving options added by the current mainline work:

| `.env` variable | Server configuration | Default |
| --- | --- | --- |
| `FISSION_CUDA_PAGE_LOCKED_DECODE_LOGITS` | `Fission:CudaPageLockedDecodeLogits` | `false` |
| `FISSION_CUDA_POOL_MAX_RETAINED_BYTES` | `Fission:CudaPoolMaxRetainedBytes` | `268435456` |
| `FISSION_CUDA_POOL_MAX_RETAINED_BUFFERS_PER_SIZE` | `Fission:CudaPoolMaxRetainedBuffersPerSize` | `8` |
| `FISSION_SAMPLED_TOKEN_IDS_OUTPUT` | `Fission:SampledTokenIdsOutput` | empty/off |

`FISSION_SAMPLED_TOKEN_IDS_OUTPUT` is for a graph that has been rewritten to expose the graph-side greedy token-id output. Do not set it for an unmodified graph.

The pinned-logits and graph-side sampling switches remain opt-in so a model/runtime combination can be validated before making either path the default.

## Startup inference gate

The CUDA Compose profile also exposes the full-stack startup probe:

```dotenv
FISSION_STARTUP_PROBE_ENABLED=true
FISSION_STARTUP_PROBE_PROMPT=Hello
FISSION_STARTUP_PROBE_MODEL_ID=my-model
FISSION_STARTUP_PROBE_MAX_TOKENS=1
FISSION_STARTUP_PROBE_TIMEOUT_SECONDS=120
```

When enabled, HTTP serving starts only after the configured tokenizer -> scheduler/runtime -> ONNX Runtime CUDA path successfully generates a token.

For dedicated host validation outside Docker, see [NVIDIA real-model smoke](nvidia-smoke.md) and the serving benchmark runner.

## Common scheduler/runtime configuration

Both Compose services share:

| `.env` variable | Server configuration | Default |
| --- | --- | ---: |
| `FISSION_KV_PAGES` | `Fission:KvPages` | `16384` |
| `FISSION_TOKENS_PER_KV_PAGE` | `Fission:TokensPerKvPage` | `16` |
| `FISSION_MAX_BATCH_TOKENS` | `Fission:MaxBatchTokens` | `2048` |
| `FISSION_MAX_BATCH_SEQUENCES` | `Fission:MaxBatchSequences` | `128` |
| `FISSION_MAX_PREFILL_CHUNK_TOKENS` | `Fission:MaxPrefillChunkTokens` | `512` |
| `FISSION_ADMISSION_CAPACITY` | `Fission:AdmissionCapacity` | `1024` |

ASP.NET Core converts double underscores in environment names such as `Fission__KvPages` into configuration sections such as `Fission:KvPages`.

## Why there is no MPS container profile

MPS is an Apple/Metal host execution lane, not a Linux/NVIDIA container runtime. The accelerator catalog therefore models `mps` separately, but Docker Compose does not pretend that a Linux container can provide Metal/MPS acceleration.

The same rule applies to NPU providers: OpenVINO/QNN/Vitis AI container profiles should be added only when their actual host devices, native runtimes, and container pass-through requirements are implemented and tested.

## CI validation

The portable container workflow still performs deterministic health/OpenAI/load-generator smoke tests.

CUDA packaging has a separate CI path that builds `Dockerfile.cuda` on a CPU runner and verifies that the resulting image can start the .NET host. Hosted CI does **not** claim GPU inference coverage; real CUDA correctness remains the responsibility of the NVIDIA host smoke/benchmark path.

See [accelerator architecture](accelerators.md), [serving benchmark](serving-benchmark.md), and [releases and container publication](releases.md).
