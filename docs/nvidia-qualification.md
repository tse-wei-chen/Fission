# NVIDIA FP16 production qualification

The RTX 3060 serving qualification established the constrained FP16 CUDA/GQA
path as Fission's preferred NVIDIA performance candidate for the validated
SmolLM2-135M-Instruct stack. It does **not** yet make FP16 the unconditional
production default.

As of 2026-10-03, the checked-in high-concurrency/long-context suite has passed
30/30 completeness with zero failures and exact usage, the separate GPU
telemetry memory gate has passed 10/10 rows with substantial VRAM headroom, and
the observed structural CUDA-placement gate has passed for
`GroupQueryAttention`. The remaining promotion question is semantic behavior.

Keep the FP32 page-locked path available as the compatibility fallback while
the semantic gate remains open.

## Validated performance baseline

The earlier unprofiled FP16 comparison used three repetitions for every row.
All 288 requests succeeded with exact usage accounting.

| Workload | C | FP32 pinned tok/s | FP16 GQA tok/s | Delta | FP32 TPOT p50 ms | FP16 TPOT p50 ms | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpu-decode | 1 | 51.86 | 135.63 | +161.53% | 19.03 | 7.35 | -61.38% |
| gpu-decode | 4 | 91.47 | 239.61 | +161.95% | 42.95 | 16.44 | -61.72% |
| gpu-decode | 8 | 140.51 | 443.44 | +215.59% | 55.80 | 17.75 | -68.19% |
| gpu-short | 1 | 59.01 | 129.23 | +119.00% | 16.31 | 7.27 | -55.43% |
| gpu-short | 4 | 111.07 | 237.35 | +113.69% | 34.25 | 15.98 | -53.34% |
| gpu-short | 8 | 196.23 | 508.62 | +159.20% | 38.72 | 15.28 | -60.54% |

## 1. Structural CUDA gate — passed on observed partial trace

The FP16 ORT profile was summarized with `-RequireCudaOp GroupQueryAttention`
and `-PartialTrace`. The observed gate result was:

```text
GroupQueryAttention | CUDAExecutionProvider | 95847 | 0 | PASS
```

Observed evidence:

- 95,847 CUDA `GroupQueryAttention` events;
- zero non-CUDA `GroupQueryAttention` events;
- zero Memcpy execution events;
- Memcpy share: 0.00%, compared with 18.57% in the earlier FP32 partial trace;
- CUDA provider share: 99.76%;
- CPU provider share: 0.24%.

ONNX Runtime reached its event limit, so this is deliberately recorded as a
partial-trace gate. It proves CUDA placement for every observed GQA event but
does not claim that truncated later events were observed.

The resulting hot-path attribution is now compute-side rather than transfer-
side: CUDA MatMul accounts for approximately 49.97% and CUDA
GroupQueryAttention approximately 21.72% of observed summed node duration.

## 2. High-concurrency and long-context gate — passed on RTX 3060

The final three-repetition qualification completed every expected row:

```text
expected:  30
validated: 30
issues:     0
```

The run covered 1,056 measured requests with zero failures and exact usage for
every row. Long-context C=4/8/16 completed all three repetitions after the CUDA
GQA continuation-prefill safety policy and FP16 gathered-past-KV fixes landed.

The long-context path has a documented latency trade-off: singleton
continuation-prefill safety increases TPOT as concurrency rises. C=8/C=16
throughput remains approximately 102–106 output tok/s while TPOT p50 rises to
approximately 64.16/119.45 ms. This is accepted for the correctness/completeness
gate and remains a separate optimization target.

### Memory/headroom evidence — passed

A separate run with `-GpuTelemetry` completed 10/10 expected rows. Across 198
samples at 500 ms intervals, the observed peaks were:

- VRAM used: 3,479 / 12,288 MiB;
- VRAM headroom: 8,809 MiB;
- GPU utilization: 86%;
- memory-controller utilization: 26%;
- temperature: 68 C.

The dated evidence is recorded in
`docs/nvidia-qualification-results-2026-10-03.md`.

## 3. Semantic/token gate — remaining promotion gate

FP16 changes arithmetic and can legitimately change greedy choices when logits
are close. Performance success is therefore not evidence of token parity or
model-quality parity.

The fixed corpus exists at:

`benchmarks/serving/semantic-parity.json`

The low-level comparator is:

`eng/compare-serving-semantic-parity.ps1`

For the validated NVIDIA host, prefer the orchestration wrapper:

`eng/run-nvidia-semantic-parity.ps1`

It starts two isolated Fission server processes on the same CUDA device:

- FP32 baseline: original FP32 graph with page-locked full-logits decode;
- FP16 candidate: FP16 graph with graph-side sampled token IDs.

It waits for both servers to become healthy, executes the checked-in semantic
corpus through the token-gated internal generation endpoint, preserves raw token
IDs and decoded text, writes JSON/Markdown reports, and cleans up both servers.
The dual-server run is semantic evidence only; do not treat latency from this
configuration as a serving benchmark.

Example:

```powershell
pwsh ./eng/run-nvidia-semantic-parity.ps1 `
  -BaselineModelPath artifacts/models/SmolLM2-135M-Instruct/onnx/model.onnx `
  -CandidateModelPath artifacts/models/SmolLM2-135M-Instruct/onnx/model.fp16.fission-greedy.onnx `
  -TokenizerPath artifacts/models/SmolLM2-135M-Instruct/tokenizer.json `
  -ModelId SmolLM2-135M-Instruct `
  -NumHiddenLayers 30 `
  -NumKvHeads 3 `
  -HeadDim 64 `
  -VocabularySize 49152 `
  -EosTokenIds 2 `
  -CudaRuntimeLibraryPath 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64\cudart64_13.dll' `
  -CandidateSampledTokenIdsOutput fission_sampled_token_ids
```

The default gate records exact-token matches, exact-text matches, common token
prefix lengths/ratios, first divergence indices, full token IDs, and decoded
text. It does not fail merely because FP16 diverges from FP32. Review divergence
case by case and distinguish exact token parity from application-level semantic
acceptability. Use `-RequireExactTokens` only when intentionally running the
stricter all-cases-exact experiment.

## Promotion rule

Promote FP16 from preferred candidate to the primary NVIDIA serving path only
when:

- the observed structural GQA CUDA gate has passed — **passed for the validated
  RTX 3060 tuple, with the documented partial-trace limitation**;
- the checked-in qualification manifest remains complete with no failures and
  exact accounting — **passed**;
- the memory/headroom telemetry remains within the validated envelope —
  **passed**;
- semantic/token comparison has been reviewed;
- the FP32 page-locked path remains selectable as a compatibility fallback.

After promotion, continue to treat new GPUs, ORT versions, CUDA versions, model
families, and precision formats as separate qualification targets rather than
assuming this RTX 3060 result transfers automatically.
