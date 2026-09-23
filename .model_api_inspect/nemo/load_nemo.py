#!/usr/bin/env python3
"""Load the .nemo checkpoint that ships in this repository, on a stock NeMo install.

    import sys
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download("bodhan-ai/indic-transcribe-core")
    sys.path.insert(0, f"{model_dir}/nemo")

    from load_nemo import load_nemo_model
    model = load_nemo_model(model_dir)
    print(model.transcribe(["audio.wav"], source_lang="hi", target_lang="hi", pnc="yes")[0].text)

The checkpoint's config names the tokenizer class it was trained with, which is not part of a
stock NeMo install, so ``load_nemo_model`` registers the copy in this folder before restoring.
Nothing inside NeMo is modified. Everything here is found relative to this file, so the same
folder works unchanged in any of our repositories.
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_MODULE = "nemo.collections.common.tokenizers.canary_multilingual_tokenizer"
TOKENIZER_FILE = os.path.join(HERE, "canary_multilingual_tokenizer.py")

__all__ = ["load_nemo_model", "register_tokenizer", "find_checkpoint"]


def register_tokenizer(path: str = TOKENIZER_FILE):
    """Make the class the checkpoint config names importable. Safe to call more than once."""
    if TOKENIZER_MODULE in sys.modules:
        return sys.modules[TOKENIZER_MODULE].CanaryMultilingualTokenizer
    spec = importlib.util.spec_from_file_location(TOKENIZER_MODULE, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[TOKENIZER_MODULE] = module
    spec.loader.exec_module(module)
    return module.CanaryMultilingualTokenizer


def find_checkpoint(model_dir: str | None = None) -> str:
    """The .nemo file in this folder (or in ``model_dir``/nemo)."""
    for d in ([os.path.join(model_dir, "nemo"), model_dir] if model_dir else []) + [HERE]:
        found = sorted(glob.glob(os.path.join(d, "*.nemo")))
        if found:
            return found[0]
    raise FileNotFoundError(f"no .nemo checkpoint found in {model_dir or HERE}")


def load_nemo_model(model_dir: str | None = None, map_location: str | None = None):
    """Register the tokenizer, restore the checkpoint, return the NeMo model."""
    register_tokenizer()
    from nemo.collections.asr.models import EncDecMultiTaskModel

    checkpoint = find_checkpoint(model_dir)
    if map_location is None:
        import torch

        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    return EncDecMultiTaskModel.restore_from(checkpoint, map_location=map_location).eval()
