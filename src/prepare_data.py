"""
Prepare a Marathi ASR dataset into NeMo manifest format.

Downloads Google FLEURS Marathi (config `mr_in`) — ungated, already 16kHz mono,
small enough to fit a near-full disk — writes each split to wav on disk and emits
a NeMo-style manifest (one JSON per line). Manifest fields follow the Canary /
multitask convention the Bodhan model inherits from `nvidia/canary-1b-v2`:

    {"audio_filepath", "duration", "text", "source_lang", "target_lang", "pnc"}

For plain ASR, source_lang == target_lang == "mr" and pnc="yes" (keep casing/
punctuation as-is). Swap DATASET_ID/CONFIG for Common Voice to scale up later.

Usage:
    python src/prepare_data.py --out data/fleurs_mr --max-train 800 --max-val 200
"""
import argparse
import json
import os

import soundfile as sf
from datasets import load_dataset

DATASET_ID = "google/fleurs"
CONFIG = "mr_in"          # Marathi (India)
LANG = "mr"
TARGET_SR = 16000


def dump_split(ds, split_dir, manifest_path, limit):
    """Write clips to wav and one manifest line each. Returns count written."""
    os.makedirs(split_dir, exist_ok=True)
    n = 0
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for ex in ds:
            if limit and n >= limit:
                break
            audio = ex["audio"]
            wav = audio["array"]
            sr = audio["sampling_rate"]
            # FLEURS is already 16k mono; assert instead of silently resampling,
            # so a future dataset swap fails loudly rather than shipping bad audio.
            assert sr == TARGET_SR, f"expected {TARGET_SR}Hz, got {sr}Hz — add a resample step"
            text = (ex.get("transcription") or ex.get("raw_transcription") or "").strip()
            if not text:
                continue
            fp = os.path.join(split_dir, f"{LANG}_{n:05d}.wav")
            sf.write(fp, wav, sr)
            duration = len(wav) / sr
            mf.write(json.dumps({
                "audio_filepath": os.path.abspath(fp),
                "duration": round(duration, 3),
                "text": text,
                "source_lang": LANG,
                "target_lang": LANG,
                "pnc": "yes",
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fleurs_mr")
    ap.add_argument("--max-train", type=int, default=800)
    ap.add_argument("--max-val", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading {DATASET_ID}:{CONFIG} (this streams a few hundred MB the first time)...")
    train = load_dataset(DATASET_ID, CONFIG, split="train")
    val = load_dataset(DATASET_ID, CONFIG, split="validation")

    n_tr = dump_split(train, os.path.join(args.out, "train"),
                      os.path.join(args.out, "train_manifest.json"), args.max_train)
    n_va = dump_split(val, os.path.join(args.out, "val"),
                      os.path.join(args.out, "val_manifest.json"), args.max_val)
    print(f"Wrote {n_tr} train / {n_va} val clips under {args.out}")
    print(f"Manifests: {args.out}/train_manifest.json , {args.out}/val_manifest.json")


if __name__ == "__main__":
    main()
