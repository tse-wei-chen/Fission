# NVIDIA FP16 qualification evidence — 2026-10-03

This note records the first high-concurrency/extended-decode qualification run
for the validated RTX 3060 + SmolLM2-135M-Instruct FP16 CUDA/GQA path.

The run was unprofiled, used three repetitions per emitted row, reported zero
failed requests, and reported exact usage for every emitted repetition.

## Observed rows

| Workload | C | Max tokens | Output tok/s | Req/s | Req/s sd | TTFT p50 ms | TPOT p50 ms | E2E p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpu-decode-extended | 8 | 256 | 392.75 | 1.53 | 0.08 | 59.65 | 20.35 | 5518.45 |
| gpu-decode-extended | 16 | 256 | 567.44 | 2.22 | 0.09 | 86.93 | 27.98 | 7536.50 |
| gpu-decode-extended | 32 | 256 | 977.70 | 3.82 | 0.75 | 91.44 | 33.95 | 8750.97 |
| gpu-short-high-concurrency | 8 | 32 | 478.27 | 14.95 | 0.72 | 38.33 | 16.04 | 572.88 |
| gpu-short-high-concurrency | 16 | 32 | 883.57 | 27.61 | 0.54 | 50.29 | 16.89 | 606.34 |
| gpu-short-high-concurrency | 32 | 32 | 1216.15 | 38.00 | 5.96 | 80.23 | 25.16 | 876.61 |
| gpu-long-context | 1 | 64 | 110.68 | 1.73 | 0.03 | 53.60 | 8.21 | 615.55 |

From C=8 to C=32, output-token throughput scales by approximately 2.49x for
the 256-token decode workload and 2.54x for the short workload. Relative to
perfect 4x linear scaling, that is roughly 62–64% scaling efficiency. TPOT rises
as concurrency increases, which is the expected throughput/latency trade-off
rather than a throughput collapse.

C=32 variability is materially higher than C=8/16: request-throughput
coefficient of variation is approximately 19.6% for extended decode and 15.7%
for the short workload. Memory headroom and repeated-run stability therefore
remain qualification concerns.

## Long-context gate remains open

The checked-in manifest requires `gpu-long-context` at C=1/4/8/16, but the
supplied report contains only C=1. The manifest itself is correct; therefore
this report must be treated as partial qualification evidence rather than a
complete pass.

Fission now provides `eng/validate-serving-suite-results.ps1` and
`eng/run-nvidia-qualification.ps1` so missing workload/concurrency/repetition
artifacts cannot silently look like a complete qualification report.

## Next gates

1. rerun the qualification wrapper and require all manifest rows;
2. run once with `-GpuTelemetry` to capture peak VRAM/headroom and utilization;
3. complete the CUDA-only `GroupQueryAttention` structural profile assertion;
4. compare FP32 and FP16 deterministic generations with the checked-in semantic
   parity corpus using raw generated token IDs.

Do not remove the FP32 page-locked compatibility path until these gates close.
