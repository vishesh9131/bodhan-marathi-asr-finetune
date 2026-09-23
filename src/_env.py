"""Import this FIRST in any script that pulls in NeMo / torchaudio.

This machine's torch is a custom 2.14.0+cu126 build (in ~/.local); the torchaudio
next to it was compiled against CUDA 12.8. torchaudio's import-time check refuses
that 12.6-vs-12.8 mismatch and raises, even though the ops are runtime-compatible
across a CUDA minor version (verified: MelSpectrogram runs on GPU fine). No public
torchaudio build matches torch 2.14, so we can't just "install the matching one".

Spoofing the reported CUDA string before torchaudio loads lets it import. This is
cosmetic — it changes a version string used only for that guard, not the runtime.
"""
import torch

if torch.version.cuda == "12.6":
    torch.version.cuda = "12.8"  # ponytail: cosmetic guard bypass; drop if torch/torchaudio ever match
