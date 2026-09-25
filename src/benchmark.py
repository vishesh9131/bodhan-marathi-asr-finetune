"""Benchmark fine-tuned Marathi ASR inference at several batch sizes.

Measures the deployment trade-off from one repeatable workload: a fixed subset
of held-out FLEURS Marathi clips.  Results include quality (WER), request
latency, throughput relative to audio duration, and peak memory allocated by
PyTorch.

Example:
  CUDA_VISIBLE_DEVICES=1 /home/vishesh/miniconda3/envs/bodhan-asr/bin/python \
    src/benchmark.py --data data/fleurs_mr \
    --finetuned outputs/indic_transcribe_mr.nemo --limit 100
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import _env  # noqa: F401 -- must precede nemo/torchaudio; see src/_env.py
import torch
from huggingface_hub import snapshot_download
from jiwer import wer

MODEL_ID = "bodhan-ai/indic-transcribe-flex"


def load_model(model_dir, nemo_path):
    sys.path.insert(0, os.path.join(model_dir, "nemo"))
    from load_nemo import register_tokenizer
    from nemo.collections.asr.models import EncDecMultiTaskModel

    register_tokenizer()
    return EncDecMultiTaskModel.restore_from(nemo_path, map_location="cuda").eval()


def read_manifest(path, limit):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return rows[:limit] if limit else rows


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100
    lo, hi = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def run_batch(model, files, batch_size):
    torch.cuda.empty_cache()
    # One warm-up avoids reporting CUDA graph/kernel setup as serving latency.
    model.transcribe(files[:1], source_lang="mr", target_lang="mr", pnc="yes", batch_size=1)
    torch.cuda.synchronize()
    # Reset *after* warm-up. Otherwise batch 1 gets charged for one-time CUDA
    # allocations while later configurations do not, which is not a fair serving
    # memory comparison.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_gib = torch.cuda.memory_allocated() / 2**30

    predictions, request_times = [], []
    started = time.perf_counter()
    for start in range(0, len(files), batch_size):
        request = files[start:start + batch_size]
        torch.cuda.synchronize()
        request_started = time.perf_counter()
        output = model.transcribe(request, source_lang="mr", target_lang="mr", pnc="yes",
                                  batch_size=batch_size)
        torch.cuda.synchronize()
        request_times.append(time.perf_counter() - request_started)
        predictions.extend((getattr(item, "text", item) or "").strip() for item in output)
    elapsed = time.perf_counter() - started
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    return predictions, elapsed, request_times, baseline_gib, peak_gib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fleurs_mr")
    parser.add_argument("--finetuned", default="outputs/indic_transcribe_mr.nemo")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--out", default="artifacts/benchmark.csv")
    args = parser.parse_args()

    if not os.path.exists(args.finetuned):
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {args.finetuned}")
    rows = read_manifest(os.path.join(args.data, "val_manifest.json"), args.limit)
    files, refs = [row["audio_filepath"] for row in rows], [row["text"] for row in rows]
    total_audio_s = sum(float(row["duration"]) for row in rows)
    model_dir = args.model_dir or snapshot_download(MODEL_ID, allow_patterns=["nemo/*"])
    model = load_model(model_dir, args.finetuned)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result_rows = []
    for batch_size in args.batches:
        print(f"Benchmarking batch_size={batch_size} on {len(files)} clips...", flush=True)
        hypotheses, elapsed, requests, baseline_gib, peak_gib = run_batch(model, files, batch_size)
        row = {
            "model": "fine-tuned decoder-only (600 steps)",
            "clips": len(files),
            "audio_seconds": round(total_audio_s, 3),
            "batch_size": batch_size,
            "wer": round(wer(refs, hypotheses), 4),
            "total_wall_seconds": round(elapsed, 3),
            "request_p50_seconds": round(percentile(requests, 50), 3),
            "request_p95_seconds": round(percentile(requests, 95), 3),
            "throughput_audio_seconds_per_wall_second": round(total_audio_s / elapsed, 3),
            "real_time_factor": round(elapsed / total_audio_s, 4),
            "model_torch_allocated_gib": round(baseline_gib, 3),
            "peak_torch_allocated_gib": round(peak_gib, 3),
            "incremental_request_memory_gib": round(peak_gib - baseline_gib, 3),
        }
        result_rows.append(row)
        print(json.dumps(row), flush=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result_rows[0].keys())
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
