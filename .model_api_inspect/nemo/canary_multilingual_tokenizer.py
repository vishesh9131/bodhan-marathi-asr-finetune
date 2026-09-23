# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Modifications Copyright (c) 2026, Bodhan AI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The tokenizer class that bodhan-ai Indic-Transcribe .nemo checkpoints are configured with.

The checkpoint's aggregate tokenizer has two sub-tokenizers, ``spl_tokens`` (the prompt
tokens) and ``multilingual`` (the text of every language). Stock NeMo's ``CanaryTokenizer``
routes text by the language id, knows only en/es/fr/de as aliases, and therefore raises
``RuntimeError: Unsupported language: 'ml'`` on our checkpoints. This subclass routes every
language the checkpoint was trained on to ``multilingual``, and keeps ``<|...|>`` tokens
inside the text tokenized as single special tokens.

It subclasses NeMo's own CanaryTokenizer on purpose: NeMo's prompt formatter branches on
``isinstance(tokenizer, CanaryTokenizer)``, so anything that is not a subclass falls into
the plain-SentencePiece branch and fails with
``AggregateTokenizer.token_to_id() missing 1 required positional argument: 'lang_id'``.

Usage with a stock NeMo install, without patching NeMo -- run this ONCE before restoring:

    import importlib.util, sys
    _t = "nemo.collections.common.tokenizers.canary_multilingual_tokenizer"
    _s = importlib.util.spec_from_file_location(_t, "canary_multilingual_tokenizer.py")
    sys.modules[_t] = importlib.util.module_from_spec(_s); _s.loader.exec_module(sys.modules[_t])

    from nemo.collections.asr.models import EncDecMultiTaskModel
    model = EncDecMultiTaskModel.restore_from("indic_transcribe_core.nemo")
    print(model.transcribe(["audio.wav"], source_lang="hi", target_lang="hi", pnc="yes")[0].text)

Alternatively copy this file into your NeMo install, next to ``canary_tokenizer.py``.
"""
import re
from typing import Dict, List

from nemo.collections.common.tokenizers.canary_tokenizer import (
    CANARY_EOS,
    CANARY_SPECIAL_TOKENIZER,
    CanaryTokenizer,
)

__all__ = ["CanaryMultilingualTokenizer"]

#: Every language this checkpoint line is trained on. All of them share the single
#: ``multilingual`` sub-tokenizer, so text in any of them is routed there.
MULTILINGUAL_LANGS = (
    "as", "bgc", "bhb", "bho", "bn", "brx", "doi", "en", "gu", "hi", "hne", "kn", "kok",
    "ks", "mai", "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur",
)

_SPECIAL_TOKEN_RE = re.compile(r"<\|.+?\|>")


class CanaryMultilingualTokenizer(CanaryTokenizer):
    """CanaryTokenizer whose per-language text all goes through one shared sub-tokenizer."""

    def __init__(self, tokenizers: Dict):
        super().__init__(tokenizers)

    def _route(self, lang_id: str) -> str:
        if lang_id in self.langs:            # a real sub-tokenizer of this checkpoint
            return lang_id
        if lang_id in MULTILINGUAL_LANGS and "multilingual" in self.langs:
            return "multilingual"
        return lang_id                        # let the base class raise its own error

    def _text_with_special_tokens_to_ids(self, text: str, lang_id: str) -> List[int]:
        """Tokenize text that may contain ``<|...|>`` tokens, preserving order."""
        parts = _SPECIAL_TOKEN_RE.split(text)
        specials = _SPECIAL_TOKEN_RE.findall(text)
        ids: List[int] = []
        for i, part in enumerate(parts):
            part = part.lstrip()
            if part:
                ids.extend(super(CanaryTokenizer, self).text_to_ids(part, lang_id))
            if i < len(specials):
                try:
                    ids.extend(self._tokenize_special_prompt(specials[i]))
                except KeyError:
                    pass                      # unknown special token: drop it, as in training
        return ids

    def text_to_ids(self, text: str, lang_id: str) -> List[int]:
        if lang_id == CANARY_SPECIAL_TOKENIZER:
            return self._tokenize_special_prompt(text)
        lang_id = self._route(lang_id)
        if text.endswith(CANARY_EOS):
            return self._text_with_special_tokens_to_ids(text[: -len(CANARY_EOS)], lang_id) + [self.eos_id]
        return self._text_with_special_tokens_to_ids(text, lang_id)
