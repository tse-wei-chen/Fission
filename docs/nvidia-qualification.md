# NVIDIA FP16 production qualification

The RTX 3060 serving qualification established the constrained FP16 CUDA/GQA
path as Fission's preferred NVIDIA performance candidate for the validated
SmolLM2-135M-Instruct stack. It does **not** yet make FP16 the unconditional
production default.

As of 2026-10-03, the checked-in high-concurrency/long-context suite has passed
30/30 completeness with zero failures and exact usage, and the separate GPU
telemetry memory gate has passed 10/10 rows with substantial VRAM headroom.
The remaining promotion questions are structural CUDA placement and semantic
behavior.

The production qualification phase closes four separate questions:

1. **structural placement** — the attention kernel must remain on CUDA;
2. **high-concurrency scaling** — useful throughput must continue beyond the
   initial C=8 smoke range without failure or pathological latency growth;
3. **longer decode/context behavior** — the path must survive larger KV-cache
   frontiers, with any singleton-safety latency cost recorded explicitly;
4. **semantic behavior** — FP16 greedy output must be evaluated on a stable
   prompt corpus before claiming parity with FP32.

Keep the FP32 page-locked path available as the compatibility fallback while
the remaining promotion gates are open.

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

TTFT also improved by roughly 55–69% across the six matched rows. The effect is
large and consistent enough that qualification, not another small decode-hot-
path experiment, remains the correct promotion workflow.

## 1. Structural CUDA gate — still open

Use a single ORT-profiled run for attribution. Profiling adds overhead, so do
not use this run as a throughput comparison.

After the profile is produced, require exclusive CUDA placement for
`GroupQueryAttention`:

```powershell
pwsh ./eng/summarize-ort-profile.ps1 `
  -ProfilePath <ort-profile.json> `
  -MarkdownPath <ort-profile-summary.md> `
  -JsonPath <ort-profile-summary.json> `
  -RequireCudaOp GroupQueryAttention
```

For an ORT trace that reports the event limit was reached, also pass
`-PartialTrace`. A partial trace can still prove that observed GQA events were
CUDA-only, but it cannot prove that an unobserved later event would not fall
back. Prefer a complete trace when practical.

Pass criteria:

- at least one observed CUDA `GroupQueryAttention` event;
- zero observed non-CUDA `GroupQueryAttention` events;
- materially lower host Memcpy share than the FP32 partial trace where Memcpy
  accounted for 18.57% of summed node duration.

## 2. High-concurrency and long-context gate — passed on RTX 3060

Use the checked-in qualification manifest:

`benchmarks/serving/workloads.gpu-qualification.json`

It contains three workloads:

- `gpu-short-high-concurrency`: C=8/16/32, 32 output tokens;
- `gpu-decode-extended`: C=8/16/32, 256 output tokens;
- `gpu-long-context`: a substantially larger prompt at C=1/4/8/16 with 64
  output tokens.

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
continuation-prefill safety increases TPOT as concurrency rises. In the final
three-run result, C=8/C=16 throughput remains approximately 102–106 output
tok/s while TPOT p50 rises to approximately 64.16/119.45 ms. This is accepted
for the correctness/completeness gate because every request completes and
accounting remains exact. It remains a separate optimization target.

The qualification command remains:

```powershell
pwsh ./eng/run-nvidia-serving-benchmark.ps1 `
  -ModelPath artifacts/models/SmolLM2-135M-Instruct/onnx/model.fp16.fission-greedy.onnx `
  -TokenizerPath artifacts/models/SmolLM2-135M-Instruct/tokenizer.json `
  -ModelId SmolLM2-135M-Instruct `
  -ModelPrecision fp16 `
  -NumHiddenLayers 30 `
  -NumKvHeads 3 `
  -HeadDim 64 `
  -VocabularySize 49152 `
  -EosTokenIds 2 `
  -SampledTokenIdsOutput fission_sampled_token_ids `
  -CudaRuntimeLibraryPath 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64\cudart64_13.dll' `
  -Manifest benchmarks/serving/workloads.gpu-qualification.json `
  -Label fission-cuda-fp16-qualification `
  -Repetitions 3
```

Do not add `-OrtProfile` or `-PageLockedDecodeLogits` to this performance run.

### Memory/headroom evidence — passed

A separate run with `-GpuTelemetry` completed 10/10 expected rows. Across 198
samples at 500 ms intervals, the observed peaks were:

- VRAM used: 3,479 / 12,288 MiB;
- VRAM headroom: 8,809 MiB;
- GPU utilization: 86%;
- memory-controller utilization: 26%;
- temperature: 68 C.

This closes the memory-headroom concern for the validated RTX 3060 tuple. The
telemetry run is supporting capacity evidence, not a replacement for the
three-repetition performance run.

The dated evidence is recorded in
`docs/nvidia-qualification-results-2026-10-03.md`.

## 3. Semantic/token gate — still open

FP16 changes arithmetic and can legitimately change greedy choices when logits
are close. Performance success is therefore not evidence of token parity or
model-quality parity.

The fixed corpus already exists at:

`benchmarks/serving/semantic-parity.json`

and the comparator is:

`eng/compare-serving-semantic-parity.ps1`

The corpus includes short factual completion, instruction following,
code/text formatting, longer contextual reasoning, and longer generation.
The comparator preserves raw token IDs and decoded text from both endpoints and
reports exact-token matches plus common-prefix divergence.

Run the same tokenizer, chat template, EOS configuration, prompt text, and
maximum output length against FP32 and FP16. Report exact token-prefix equality
separately from application-level acceptability; do not require bitwise/logit
equality.

## Promotion rule

Promote FP16 from preferred candidate to the primary NVIDIA serving path only
when:

- the structural GQA CUDA gate passes;
- the checked-in qualification manifest remains complete with no failures and
  exact accounting — **passed for the validated RTX 3060 tuple**;
- the memory/headroom telemetry remains within the validated envelope —
  **passed for the validated RTX 3060 tuple**;
- semantic/token comparison has been reviewed;
- the FP32 page-locked path remains selectable as a compatibility fallback.

After promotion, continue to treat new GPUs, ORT versions, CUDA versions, model
families, and precision formats as separate qualification targets rather than
assuming this RTX 3060 result transfers automatically.
