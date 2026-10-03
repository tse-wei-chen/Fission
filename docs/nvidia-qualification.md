# NVIDIA FP16 production qualification

The RTX 3060 three-repetition serving benchmark established the constrained
FP16 CUDA/GQA path as Fission's preferred NVIDIA performance candidate for the
validated SmolLM2-135M-Instruct stack. It does **not** yet make FP16 the
unconditional production default.

The production qualification phase closes four separate questions:

1. **structural placement** — the attention kernel must remain on CUDA;
2. **high-concurrency scaling** — useful throughput must continue beyond the
   initial C=8 smoke range without failure or pathological latency growth;
3. **longer decode/context behavior** — the win must survive larger KV-cache
   frontiers;
4. **semantic behavior** — FP16 greedy output must be evaluated on a stable
   prompt corpus before claiming parity with FP32.

Keep the FP32 page-locked path available as the compatibility fallback while
these gates are open.

## Validated performance baseline

The unprofiled FP16 run used three repetitions for every row. All 288 requests
succeeded with exact usage accounting.

| Workload | C | FP32 pinned tok/s | FP16 GQA tok/s | Delta | FP32 TPOT p50 ms | FP16 TPOT p50 ms | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpu-decode | 1 | 51.86 | 135.63 | +161.53% | 19.03 | 7.35 | -61.38% |
| gpu-decode | 4 | 91.47 | 239.61 | +161.95% | 42.95 | 16.44 | -61.72% |
| gpu-decode | 8 | 140.51 | 443.44 | +215.59% | 55.80 | 17.75 | -68.19% |
| gpu-short | 1 | 59.01 | 129.23 | +119.00% | 16.31 | 7.27 | -55.43% |
| gpu-short | 4 | 111.07 | 237.35 | +113.69% | 34.25 | 15.98 | -53.34% |
| gpu-short | 8 | 196.23 | 508.62 | +159.20% | 38.72 | 15.28 | -60.54% |

TTFT also improved by roughly 55–69% across the six matched rows. The effect is
large and consistent enough that the next work is qualification rather than
another small decode-hot-path experiment.

## 1. Structural CUDA gate

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

## 2. High-concurrency and long-context gate

Use the checked-in qualification manifest:

`benchmarks/serving/workloads.gpu-qualification.json`

It contains three workloads:

- `gpu-short-high-concurrency`: C=8/16/32, 32 output tokens;
- `gpu-decode-extended`: C=8/16/32, 256 output tokens;
- `gpu-long-context`: a substantially larger prompt at C=1/4/8/16 with 64
  output tokens.

Run it unprofiled with three repetitions:

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

Pass criteria are intentionally behavioral rather than a fixed absolute tok/s
threshold:

- zero failed requests;
- exact usage for every repetition;
- no server crash, CUDA allocation failure, stale-KV failure, or timeout;
- throughput should plateau gracefully as concurrency rises rather than
  collapse while latency grows without useful throughput;
- the 256-token decode workload must complete at C=32;
- the long-context workload must complete through C=16.

Record the resulting report rather than encoding RTX-3060-specific timing as a
hosted-CI threshold.

## 3. Semantic/token gate

FP16 changes arithmetic and can legitimately change greedy choices when logits
are close. Performance success is therefore not evidence of token parity or
model-quality parity.

Use a fixed prompt corpus that includes:

- short factual completion;
- instruction following;
- code/text formatting;
- a longer contextual question;
- prompts that generate at least 64 tokens.

Run the same tokenizer, chat template, EOS configuration, prompt text, and
maximum output length against FP32 and FP16. Persist raw generated token IDs and
text. Report exact token-prefix equality separately from application-level
acceptability; do not require bitwise/logit equality.

This semantic gate should be implemented as a reproducible artifact before
promoting FP16 to the unconditional default.

## Promotion rule

Promote FP16 from preferred candidate to the primary NVIDIA serving path only
when:

- the structural GQA CUDA gate passes;
- the checked-in qualification manifest completes with no failures and exact
  accounting;
- semantic/token comparison has been reviewed;
- the FP32 page-locked path remains selectable as a compatibility fallback.

After promotion, continue to treat new GPUs, ORT versions, CUDA versions, model
families, and precision formats as separate qualification targets rather than
assuming this RTX 3060 result transfers automatically.