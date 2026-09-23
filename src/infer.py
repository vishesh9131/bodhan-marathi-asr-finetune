"""
Transcribe Marathi clips and report WER, comparing the pretrained model against a
fine-tuned checkpoint. Metrics aren't graded — this just demonstrates the loop
closes end-to-end and prints a before/after so the training run is legible.

Usage:
  python src/infer.py --data data/fleurs_mr --finetuned outputs/indic_transcribe_mr.nemo
  python src/infer.py --data data/fleurs_mr            # base model only
"""
import argparse
import json
import os
import sys

import _env  # noqa: F401  -- must precede nemo/torchaudio; see src/_env.py
from huggingface_hub import snapshot_download
from jiwer import wer

MODEL_ID = "bodhan-ai/indic-transcribe-flex"


def load(model_dir, nemo_path=None):
    sys.path.insert(0, os.path.join(model_dir, "nemo"))
    from load_nemo import load_nemo_model, register_tokenizer
    if nemo_path:
        register_tokenizer()
        from nemo.collections.asr.models import EncDecMultiTaskModel
        return EncDecMultiTaskModel.restore_from(nemo_path, map_location="cuda").eval()
    return load_nemo_model(model_dir)


def read_manifest(path, limit):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return rows[:limit] if limit else rows


def transcribe(model, files):
    outs = model.transcribe(files, source_lang="mr", target_lang="mr", pnc="yes", batch_size=8)
    return [(getattr(o, "text", o) or "").strip() for o in outs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fleurs_mr")
    ap.add_argument("--finetuned", default=None, help="path to fine-tuned .nemo")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--model-dir", default=None)
    args = ap.parse_args()

    model_dir = args.model_dir or snapshot_download(MODEL_ID, allow_patterns=["nemo/*"])
    rows = read_manifest(os.path.join(args.data, "val_manifest.json"), args.limit)
    files = [r["audio_filepath"] for r in rows]
    refs = [r["text"] for r in rows]

    base = load(model_dir)
    base_hyp = transcribe(base, files)
    print(f"Base model     WER: {wer(refs, base_hyp):.3f}  ({len(files)} clips)")

    if args.finetuned and os.path.exists(args.finetuned):
        ft = load(model_dir, args.finetuned)
        ft_hyp = transcribe(ft, files)
        print(f"Fine-tuned     WER: {wer(refs, ft_hyp):.3f}")

    # a couple of qualitative examples
    for i in range(min(3, len(files))):
        print(f"\nref : {refs[i]}")
        print(f"base: {base_hyp[i]}")
        if args.finetuned and os.path.exists(args.finetuned):
            print(f"ft  : {ft_hyp[i]}")


if __name__ == "__main__":
    main()
