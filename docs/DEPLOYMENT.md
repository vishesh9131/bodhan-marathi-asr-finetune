# Deployment benchmark

This experiment turns the fine-tuned Marathi ASR checkpoint into a deployment
decision rather than reporting WER alone. It measures the same 100 held-out
FLEURS Marathi clips at batch sizes 1, 2, 4, and 8 on one RTX A6000. The
checkpoint and evaluation clips are held fixed; only serving batch size changes.

## Method

- **Model:** decoder-only fine-tuned `indic-transcribe-flex` checkpoint (600
  steps), as described in `APPROACH.md`.
- **Hardware:** one NVIDIA RTX A6000 (48 GB). The GPU was isolated with
  `CUDA_VISIBLE_DEVICES` to avoid competing training workloads.
- **Quality:** WER against the existing held-out transcript references.
- **Latency:** each call to `model.transcribe` is one request. The benchmark
  reports p50/p95 request latency after a one-clip warm-up; CUDA is synchronized
  around each request so timings reflect device work.
- **Throughput:** total audio seconds divided by wall-clock seconds. A real-time
  factor below 1 means audio is processed faster than its duration.
- **Memory:** PyTorch reports the warmed model allocation, peak allocation, and
  their difference during measured requests. This is model-process memory—not
  whole-device memory—and avoids charging one-time CUDA warm-up allocations to
  the first configuration.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=1 /home/vishesh/miniconda3/envs/bodhan-asr/bin/python \
  src/benchmark.py --data data/fleurs_mr \
  --finetuned outputs/indic_transcribe_mr.nemo --limit 100 \
  --out artifacts/benchmark.csv
```

The committed `artifacts/benchmark.csv` is the result of this command. The
script accepts `--batches` and `--limit` to repeat the study with a different
traffic profile or evaluation slice.

## Results

All settings produced the same transcript set and **0.1648 WER**; serving batch
size affects operational behavior, not this checkpoint's accuracy. The workload
contains 100 clips totaling 1,315.62 seconds of audio.

| Batch | p50 request | p95 request | Throughput | Real-time factor | Peak allocated GPU memory |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.831 s | 1.408 s | 15.191 audio s/s | 0.0658 | 4.777 GiB |
| 2 | 1.051 s | 1.595 s | 24.301 audio s/s | 0.0412 | 4.953 GiB |
| 4 | 1.363 s | 1.878 s | 39.507 audio s/s | 0.0253 | 5.304 GiB |
| 8 | 2.157 s | 3.090 s | 44.444 audio s/s | 0.0225 | 6.006 GiB |

The fine-tuned model itself uses 4.602 GiB. Batch 8 is only 12.5% faster than
batch 4 but raises p95 request latency by 65% and adds 0.702 GiB of peak memory;
batch 4 is therefore the balanced queued-workload configuration.

## Deployment decision

| Workload | Recommended batch | Reason |
|---|---:|---|
| Interactive/live transcription | 1 | Lowest request latency; each caller is served independently. |
| Small queued jobs | 4 | 39.507 audio s/s with 1.878 s p95 request latency: most of batch 8's throughput at a lower latency/memory cost. |
| Offline/bulk transcription | 8 | Highest measured throughput (44.444 audio s/s); prioritize it only when request wait time is acceptable. |

The numbers in `benchmark.csv` are deliberately reported beside the above
recommendations. This avoids treating batch size as a universal answer: a live
application optimizes p95 request latency, whereas a batch pipeline optimizes
throughput and cost per audio hour.
