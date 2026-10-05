# Container usage

Fission ships separate user-facing Compose files for portable and NVIDIA CUDA serving.

| Use case | Compose file | Env template | Image Dockerfile |
| --- | --- | --- | --- |
| Portable / deterministic / ONNX CPU | `compose.yaml` | `.env.example` | `Dockerfile` |
| NVIDIA CUDA / ONNX Runtime GPU | `compose.cuda.yaml` | `.env.cuda.example` | `Dockerfile.cuda` |

The files are intentionally standalone. Users do not need Compose profiles or multiple `-f` overlays to select an accelerator lane.

Model files are **not** baked into either image. `models/` is excluded from the Docker build context and is mounted read-only at `/models`.

## Portable/default serving

Start with the portable configuration:

```bash
cp .env.example .env
docker compose up --build -d
```

Health check:

```bash
curl --fail http://localhost:8000/healthz
```

The default composition is self-contained:

```text
Backend=deterministic
ExecutionProvider=cpu
Tokenizer=deterministic
Device=cpu:0
```

To use ONNX Runtime CPU serving instead, configure the model/tokenizer geometry in `.env` and set the backend/provider/tokenizer values accordingly.

## NVIDIA CUDA serving

The CUDA configuration is separate from the portable file so users who only need CPU/default serving do not need to understand NVIDIA-specific settings.

Requirements:

- compatible NVIDIA driver;
- NVIDIA Container Toolkit;
- a compatible decoder-with-past ONNX model and tokenizer;
- the model geometry required by `Fission.Server`.

Create the CUDA env file:

```bash
cp .env.cuda.example .env.cuda
```

Fill in at least the model-specific fields in `.env.cuda`:

```dotenv
FISSION_MODEL_DIR=./models
FISSION_MODEL_FILE=decoder_with_past_model.onnx
FISSION_TOKENIZER_FILE=tokenizer.json
FISSION_MODEL_ID=my-model
FISSION_MODEL_PRECISION=fp32
FISSION_NUM_HIDDEN_LAYERS=<layers>
FISSION_NUM_KV_HEADS=<kv-heads>
FISSION_HEAD_DIM=<head-dim>
FISSION_VOCABULARY_SIZE=<vocabulary-size>
FISSION_EOS_TOKEN_IDS=<comma-separated-eos-ids>
```

Start CUDA serving:

```bash
docker compose \
  --env-file .env.cuda \
  -f compose.cuda.yaml \
  up --build -d
```

Health check:

```bash
curl --fail http://localhost:8000/healthz
```

The CUDA Compose file requests GPU access with `gpus: all`. `FISSION_CUDA_DEVICE_ID` selects the CUDA ordinal visible inside the container, while `FISSION_DEVICE` is the Fission runtime identity and defaults to `cuda:0`.

## CUDA image

`Dockerfile.cuda` uses an NVIDIA CUDA + cuDNN runtime base and copies the .NET 10 ASP.NET runtime into that environment. The current default base is:

```text
nvidia/cuda:13.4.2-cudnn-runtime-ubuntu24.04
```

Override `FISSION_CUDA_RUNTIME_IMAGE` in `.env.cuda` when validating a different compatible environment.

## CUDA serving options

The dedicated CUDA env template exposes accelerator-specific switches without polluting the portable user configuration:

| `.env.cuda` variable | Server configuration | Default |
| --- | --- | --- |
| `FISSION_MODEL_PRECISION` | `Fission:ModelPrecision` | `fp32` |
| `FISSION_CUDA_PAGE_LOCKED_DECODE_LOGITS` | `Fission:CudaPageLockedDecodeLogits` | `false` |
| `FISSION_CUDA_POOL_MAX_RETAINED_BYTES` | `Fission:CudaPoolMaxRetainedBytes` | `268435456` |
| `FISSION_CUDA_POOL_MAX_RETAINED_BUFFERS_PER_SIZE` | `Fission:CudaPoolMaxRetainedBuffersPerSize` | `8` |
| `FISSION_SAMPLED_TOKEN_IDS_OUTPUT` | `Fission:SampledTokenIdsOutput` | empty/off |
| `FISSION_ORT_PROFILE_OUTPUT_PATH_PREFIX` | `Fission:OrtProfileOutputPathPrefix` | empty/off |
| `FISSION_CONTROL_TOKEN` | `Fission:ControlToken` | empty/off |

### FP16 CUDA GQA path

The constrained FP16 CUDA path requires graph-side sampled token IDs and does not use the page-locked full-logits path:

```dotenv
FISSION_MODEL_FILE=model.fp16.fission-greedy.onnx
FISSION_MODEL_PRECISION=fp16
FISSION_SAMPLED_TOKEN_IDS_OUTPUT=fission_sampled_token_ids
FISSION_CUDA_PAGE_LOCKED_DECODE_LOGITS=false
```

Run the repository's NVIDIA one-shot hardware gate before treating a converted graph as a serving configuration.

### ONNX Runtime profiling

ORT profiling is diagnostic and adds overhead. Keep it disabled for throughput comparisons.

The CUDA Compose file mounts `FISSION_ARTIFACTS_DIR` at `/artifacts`. To persist a profile:

```dotenv
FISSION_ARTIFACTS_DIR=./artifacts/container
FISSION_ORT_PROFILE_OUTPUT_PATH_PREFIX=/artifacts/ort-profile
FISSION_CONTROL_TOKEN=<strong-ephemeral-token>
```

Use graceful shutdown so ONNX Runtime can dispose the session and flush the profile:

```bash
curl --fail \
  -X POST \
  -H 'X-Fission-Control-Token: <strong-ephemeral-token>' \
  http://localhost:8000/internal/control/shutdown
```

When `FISSION_CONTROL_TOKEN` is empty, the internal control route is not mapped.

## Startup inference gate

Both Compose files expose the startup probe settings. For CUDA, set them in `.env.cuda`:

```dotenv
FISSION_STARTUP_PROBE_ENABLED=true
FISSION_STARTUP_PROBE_PROMPT=Hello
FISSION_STARTUP_PROBE_MODEL_ID=my-model
FISSION_STARTUP_PROBE_MAX_TOKENS=1
FISSION_STARTUP_PROBE_TIMEOUT_SECONDS=120
```

When enabled, HTTP serving starts only after the configured tokenizer -> runtime -> backend path successfully generates a token.

## Why there is no MPS Compose file

MPS is an Apple/Metal host-native execution lane rather than a Linux container runtime. Fission models MPS separately in the accelerator catalog but does not present a non-functional Docker configuration for it.

The same rule applies to NPU integrations: OpenVINO/QNN/Vitis AI Compose files should be introduced only with a concrete runtime, device pass-through story, and validation path.

## CI validation

CI validates `compose.yaml` and `compose.cuda.yaml` independently. The portable workflow performs deterministic health/OpenAI/load-generator smoke tests. CUDA packaging builds `Dockerfile.cuda`, validates the standalone CUDA Compose contract, and verifies that the .NET/Fission host starts on a CPU runner without claiming real GPU inference coverage.

Real CUDA correctness remains covered by the dedicated NVIDIA smoke, qualification, semantic-parity, and serving benchmark paths.

See [accelerator architecture](accelerators.md), [serving benchmark](serving-benchmark.md), and [releases and container publication](releases.md).
