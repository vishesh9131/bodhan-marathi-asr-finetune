#!/usr/bin/env python3
"""Transcribe audio with the .nemo checkpoint in this repository.

    pip install "nemo_toolkit[asr]"
    python nemo/inference_nemo.py audio.wav --lang hi
    python nemo/inference_nemo.py long_call.wav --lang ta        # chunked automatically
    python nemo/inference_nemo.py *.wav --lang bn --batch-size 8

Audio longer than 30 s -- the length this model trains on -- is cut at natural pauses and
decoded piece by piece (``long_form_nemo.py``), the same way the Hugging Face model does it.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_nemo import load_nemo_model          # noqa: E402
from long_form_nemo import transcribe_long_nemo  # noqa: E402

CHUNK_ABOVE_SECONDS = 30.0


def duration_seconds(path: str) -> float:
    import soundfile as sf

    info = sf.info(path)
    return info.frames / info.samplerate


def main() -> int:
    p = argparse.ArgumentParser(description="Transcribe audio with the bundled .nemo checkpoint.",
                                epilog="Audio over 30 s is chunked at natural pauses automatically.")
    p.add_argument("audio", nargs="+", help="audio file(s)")
    p.add_argument("--lang", required=True, help="language code, e.g. hi (see the README)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="pieces (or files) per batch; 8 is ~2x faster than 1 on long audio")
    p.add_argument("--device", default=None, help="cuda, cpu, ... (default: auto)")
    p.add_argument("--model-dir", default=None, help="where the .nemo lives (default: this folder)")
    args = p.parse_args()

    short, long, failures = [], [], 0
    for path in args.audio:
        if not os.path.exists(path):
            print(f"{path}: not found", file=sys.stderr)
            failures += 1
            continue
        try:
            (long if duration_seconds(path) > CHUNK_ABOVE_SECONDS else short).append(path)
        except Exception as e:                  # unreadable / corrupt file: report, keep going
            print(f"{path}: {type(e).__name__}: {e}", file=sys.stderr)
            failures += 1
    if not short and not long:
        return 1

    model = load_nemo_model(args.model_dir, map_location=args.device)
    texts: dict[str, str] = {}

    if short:                                    # whole-file decoding, batched
        outs = model.transcribe(short, batch_size=args.batch_size,
                                source_lang=args.lang, target_lang=args.lang, pnc="yes")
        for path, out in zip(short, outs):
            texts[path] = (getattr(out, "text", out) or "").strip()

    for path in long:                            # cut at pauses, then decode the pieces
        try:
            texts[path] = transcribe_long_nemo(model, path, lang=args.lang,
                                               batch_size=args.batch_size)
        except Exception as e:
            print(f"{path}: {type(e).__name__}: {e}", file=sys.stderr)
            failures += 1

    many = len(texts) > 1
    for path in args.audio:
        if path in texts:
            if many:
                print(f"=== {os.path.basename(path)}")
            print(texts[path])
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
