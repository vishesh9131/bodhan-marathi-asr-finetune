#!/usr/bin/env python3
"""Long-form transcription on the NeMo path.

The checkpoint decodes whole files and trains on clips of up to 30 s; past that a single
pass collapses into repetition. This cuts the audio at natural pauses into pieces of at
most 30 s, decodes them (batched, which the NeMo API does well), and joins the text.

The splitter, the 30 s cap and the loop guard are the ones the Hugging Face model uses --
imported from ``long_form.py`` in the parent folder rather than reimplemented, so both
paths cut audio identically. Cutting is what costs accuracy: on a 42,764-clip benchmark,
cap 10 s +1.89 OIWER, 15 s +0.85, 20 s +0.29, 25 s +0.09, 30 s +0.00. DO NOT LOWER the cap.

    import sys
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download("bodhan-ai/indic-transcribe-core")
    sys.path.insert(0, f"{model_dir}/nemo")

    from load_nemo import load_nemo_model
    from long_form_nemo import transcribe_long_nemo

    model = load_nemo_model(model_dir)
    print(transcribe_long_nemo(model, "long_audio.wav", lang="hi"))
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from long_form import (  # noqa: E402  (the model's own splitter and guard)
    LOOP_MIN_WORDS,
    LOOP_REPETITION,
    _collapse_repeats,
    _repetition,
    split_on_silence,
)

__all__ = ["transcribe_long_nemo", "load_audio"]

SAMPLE_RATE = 16000
MIN_PIECE_SAMPLES = 1600            # 0.1 s; shorter pieces carry no speech


def load_audio(audio, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Path or array -> mono float32 at 16 kHz."""
    if isinstance(audio, str):
        import soundfile as sf

        data, sr = sf.read(audio, dtype="float32", always_2d=True)
        wav = torch.as_tensor(data).mean(dim=1)
        if sr != sample_rate:
            import torchaudio

            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        return wav
    wav = torch.as_tensor(audio).float().squeeze()
    if wav.dim() == 2:                  # stereo: the shorter axis is the channels
        wav = wav.mean(dim=0 if wav.shape[0] < wav.shape[1] else 1)
    if wav.dim() != 1:
        raise ValueError(f"expected mono or stereo audio, got shape {tuple(wav.shape)}")
    return wav


def _decode(model, pieces, lang, batch_size, **kw):
    """Transcribe a list of waveforms, in batches."""
    if not pieces:
        return []
    out = model.transcribe([p.numpy() for p in pieces], batch_size=batch_size,
                           source_lang=lang, target_lang=lang, pnc="yes", **kw)
    return [(getattr(o, "text", o) or "").strip() for o in out]


def transcribe_long_nemo(model, audio, lang: str, batch_size: int = 8, **kw) -> str:
    """Transcribe audio of any length with a NeMo model. Returns one string.

    ``lang`` is required: this path has no language identification of its own, and a wrong
    language yields the wrong script rather than an obvious error.
    """
    wav = load_audio(audio)
    pieces = [wav[a:b] for a, b in split_on_silence(wav)]
    pieces = [p for p in pieces if p.numel() >= MIN_PIECE_SAMPLES]
    texts = _decode(model, pieces, lang, batch_size, **kw)

    # A piece whose output is mostly one phrase repeated has collapsed into a loop. Split it
    # at a pause and decode the halves; that changes what the decoder conditions on and
    # breaks the loop. If it still loops, collapse the repeated phrase so one stuck piece
    # cannot flood the transcript.
    for i, (piece, text) in enumerate(zip(pieces, texts)):
        words = text.split()
        if len(words) < LOOP_MIN_WORDS or _repetition(words) <= LOOP_REPETITION:
            continue
        texts[i] = _retry(model, piece, lang, batch_size, kw, depth=0)
    return " ".join(t for t in texts if t)


def _retry(model, piece, lang, batch_size, kw, depth: int) -> str:
    if depth >= 2 or piece.numel() < 4 * SAMPLE_RATE:
        return " ".join(_collapse_repeats(_decode(model, [piece], lang, 1, **kw)[0].split()))
    half = piece.numel() / SAMPLE_RATE / 2
    parts = split_on_silence(piece, target=half, hard_max=half + 2.0)
    if len(parts) < 2:                                  # no pause: cut in the middle
        m = piece.numel() // 2
        parts = [(0, m), (m, piece.numel())]
    halves = [piece[a:b] for a, b in parts if b - a >= MIN_PIECE_SAMPLES]
    out = []
    for h, text in zip(halves, _decode(model, halves, lang, batch_size, **kw)):
        words = text.split()
        looped = len(words) >= LOOP_MIN_WORDS and _repetition(words) > LOOP_REPETITION
        out.append(_retry(model, h, lang, batch_size, kw, depth + 1) if looped else text)
    return " ".join(t for t in out if t)


if __name__ == "__main__":
    from load_nemo import load_nemo_model

    if len(sys.argv) < 3:
        raise SystemExit("usage: python nemo/long_form_nemo.py <audio> <lang> [batch_size]")
    m = load_nemo_model()
    print(transcribe_long_nemo(m, sys.argv[1], sys.argv[2],
                               batch_size=int(sys.argv[3]) if len(sys.argv) > 3 else 8))
