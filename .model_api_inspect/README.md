---
extra_gated_prompt: "Please provide your details and agree to the [LICENSE](https://github.com/Bodhan-AI/bodhan-model-info/blob/main/licenses/indic-open-model-license/v1/Indic_Open_Model_License.md) [[simpler version](https://github.com/Bodhan-AI/bodhan-model-info/blob/main/licenses/indic-open-model-license/v1/Indic_Open_Model_License_Deed.md)] to request access."
extra_gated_fields:
  Company / Organization: text
  Country: country
  Intended Use Case:
    type: select
    options:
      - Research
      - Commercial
      - Education
      - label: Other
        value: other
  I agree to the license terms: checkbox
license: other
language:
- en
- as
- bn
- brx
- doi
- gu
- hi
- kn
- ks
- kok
- mai
- ml
- mni
- mr
- ne
- or
- pa
- sa
- sat
- sd
- ta
- te
- ur
- bho
- hne
- bgc
- bhb
library_name: nemo
pipeline_tag: automatic-speech-recognition
base_model: nvidia/canary-1b-v2
tags:
- automatic-speech-recognition
- speech
- audio
- asr
- multilingual
- indic
- code-switching
- code-mixing
- romanization
- transliteration
- language-identification
- nemo
- canary
- fastconformer
metrics:
- wer
- cer
---

<div align="center">
  <img src="banner.png" alt="Indic-Transcribe Flex">
</div>

<h1 id="indic-transcribe-flex" style="color:#FFD21E;">Indic-Transcribe-flex</h1>

<div align="center">

[![Model Arch](https://img.shields.io/badge/Model_Arch-FastConformer--Transformer-C1440E?style=flat#model-badge)](#model-architecture)
[![Params](https://img.shields.io/badge/Params-1.2B-C1440E?style=flat#model-badge)](#model-architecture)
[![Languages](https://img.shields.io/badge/Languages-27-C1440E?style=flat#model-badge)](#supported-languages)
[![Language](https://img.shields.io/badge/Language-Multilingual-C1440E?style=flat#model-badge)](#supported-languages)
[![License](https://img.shields.io/badge/License-Indic%20Open%20Model%20License%20v1.0-C1440E?style=flat#model-badge)](#license--terms-of-use)


</div>

**Multilingual speech recognition for 27 Indian languages with native-script, mixed-script, and romanized output.**

<div align="center">
  <img src="model-diagram.png" alt="Indic-Transcribe: 27 languages in, all output features out" width="860">
</div>

**Quick links**:  [Blog](bodhan.ai/research/blogs/indic-transcribe) · [Demo](https://youtu.be/5GuSEcs4ucc?si=xNhlzdFBWaIEW2fl) · [Try it out](bodhan.ai/developers/indic-transcribe) . [Github](https://github.com/Bodhan-AI/bodhan_genai)

**Indic-Transcribe-Flex** is a multilingual Automatic Speech Recognition (ASR) model built for **27 Indian languages**. It is trained to be robust and general-purpose: it handles the full diversity of Indian accents and holds up in noisy real-world conditions, from crowded markets to call-centre floors, with strong coverage in the domains where Indian voice products are actually built: education, agriculture, and healthcare.

Unlike traditional ASR systems that return only a native-script transcript, Indic-Transcribe offers **three transcription modes**: native script, romanized text, or true code-mixed output so you can match the output to what your product expects.

---

<h2 id="why-choose-indic-transcribe" style="color:#FFD21E;">Why Choose Indic-Transcribe?</h2>

- 🔀 **Code-mixing, natively.** Indians rarely speak one language at a time. Indic-Transcribe transcribes Hinglish and other mixed speech as it is actually spoken, instead of forcing it into a single language.
- 📝 **Three transcription modes.** *Native script* for fully native output; *mixed script* for native words in native script with English and numerals in Latin; *romanized* for everything in Latin script.
- 🏥 **Domain coverage where it matters.** Deep vocabulary in education, agriculture, and healthcare.
- 🌐 **Language identification built in.** Use the model directly as a language-ID system, or let it auto-detect the language and then transcribe.
---

<h2 id="supported-languages" style="color:#FFD21E;">Supported Languages</h2>

The model covers **27 languages** across four groups:

| Group | Languages |
| --- | --- |
| **Indian-accented English** | English benchmarked across speakers from 19 states |
| **22 constitutionally recognised languages** | Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu, Urdu |
| **Hindi dialects** | Chhattisgarhi, Haryanvi |
| **Extremely low-resource** | Bhili, Bhojpuri |

See [Supported Language Codes](#supported-language-codes) for the code to pass at inference time.

<h3 style="color:#FFD21E;">The three transcription modes</h3>

The same utterance, three renderings:

| Mode | Output |
| --- | --- |
| Native script | `मैंने कल पांच बजे तीन फाइल्स अपलोड कीं` |
| Mixed script | `मैंने कल 5 बजे 3 files upload कीं` |
| Romanized | `maine kal 5 baje 3 files upload kin` |

---

<h2 id="model-architecture" style="color:#FFD21E;">Model Architecture</h2>

**Architecture Type:** NVIDIA Canary: FastConformer encoder with a Transformer decoder.

Indic-Transcribe-Flex is built on the [nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2) architecture. The FastConformer encoder produces acoustic representations that the Transformer decoder converts into text, with task tokens selecting the transcription mode and target language.

|  |  |
| --- | --- |
| **Model name** | Indic-Transcribe-Flex |
| **Task** | Speech-to-Text (Automatic Speech Recognition) |
| **Base model** | [nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2) |
| **Total parameters** | 1.2B |
| **Encoder** | FastConformer: 32 layers, 811M params, 1024 hidden dim, 8 attention heads, conv kernel 9 |
| **Decoder** | Transformer: 24 layers, 419M params, 1024 hidden size, 8 attention heads |
| **Vocabulary** | 7,152 tokens (1,152 special / task + 6,000 multilingual) |
| **Sub-word algorithm** | BPE (byte fallback disabled) |
| **Precision** | fp16 |
| **Checkpoint size** | 4.6 GB |


---

<h2 id="results-at-a-glance" style="color:#FFD21E;">Results at a Glance</h2>

Results below are from the **Voice of India** benchmark. ASR performance is measured using Word Error Rate (WER); **lower is better**. Best score per column is in **bold**.

| Model | Average | Assamese | Bhojpuri | Bengali | Gujarati | Hindi | Chhattisgarhi | Kannada | Maithili | Malayalam | Marathi | Odia | Punjabi | Tamil | Telugu | Urdu |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Indic-Transcribe-core** | 8.7 | 8.2 | 13.3 | 4.3 | 9.2 | 3.5 | 13.6 | 7.4 | 11.3 | 11.5 | 5.7 | 8.7 | 8.3 | 9.0 | 11.4 | 5.0 |
| Saaras V3 | 10.7 | 9.1 | 17.9 | 5.2 | 9.7 | 3.8 | 14.0 | 8.8 | 14.2 | 12.2 | 6.5 | 11.1 | 8.6 | 9.1 | 13.5 | 7.5 |
| **Indic-Transcribe-flex** | 11.3 | 9.6 | 18.5 | 5.0 | 10.8 | 4.1 | 13.6 | 9.7 | 15.1 | 13.7 | 6.6 | 10.2 | 9.5 | 10.8 | 13.1 | 5.6 |
| Indic Conformer | 17.8 | 13.1 | 30.3 | 9.5 | 16.3 | 6.5 | 24.5 | 16.3 | 16.3 | 28.2 | 11.6 | 13.1 | 19.1 | 16.2 | 20.0 | 8.0 |
| Gemini 3 Pro | 21.1 | 23.7 | 24.1 | 10.3 | 18.1 | 9.3 | 19.6 | 20.1 | 27.2 | 21.0 | 14.0 | 25.7 | 19.3 | 15.5 | 24.6 | 10.6 |
| Gemini 3 Flash | 23.1 | 25.0 | 20.2 | 11.1 | 20.3 | 7.1 | 21.7 | 18.5 | 28.0 | 30.1 | 14.2 | 23.3 | 23.2 | 17.3 | 25.5 | 11.0 |
| Gemma E4B | 36.1 | 45.0 | 27.0 | 19.6 | 27.3 | 9.1 | 24.0 | 31.0 | 36.7 | 44.4 | 24.4 | 44.4 | 23.4 | 37.9 | 41.6 | 14.2 |
| OmniASR LLM 7B | 44.5 | 23.9 | 26.3 | 20.9 | 32.0 | 9.6 | 20.7 | 35.0 | 44.6 | 48.8 | 24.5 | 72.3 | 31.7 | 40.6 | 48.7 | 14.8 |
| OmniASR CTC 7B | 62.5 | 33.3 | 39.2 | 43.5 | 62.5 | 20.7 | 33.4 | 50.0 | 51.6 | 61.3 | 33.6 | 90.2 | 80.7 | 58.3 | 62.1 | 89.3 |

Indic-Transcribe Flex offers support for more transcription modes and languages. Do check out [Indic-Transcribe Core](https://huggingface.co/bodhan-ai/indic-transcribe-core) for better accuracy.

<!-- ---

<h2 id="throughput--efficiency" style="color:#FFD21E;">Throughput & Efficiency</h2>

Measured on a single NVIDIA H100, fp16.

| Metric | Indic-Transcribe-Flex | Indic-Transcribe-lite |
| --- | :---: | :---: |
| Latency | 100 ms | 30 ms |
| RTFx | 500 | 1000 |
| Peak GPU memory | *TBD* | *TBD* |
| Checkpoint size | 4.6 GB | *TBD* |

> RTFx is the inverse real-time factor: RTFx 500 means one hour of audio is transcribed in ~7.2 seconds. -->

---

<h2 id="how-to-use-this-model" style="color:#FFD21E;">How to Use this Model</h2>

<h3 style="color:#FFD21E;">Installation</h3>

```bash
pip install torch torchaudio transformers sentencepiece soundfile
```

The model code ships inside this repository, so there is nothing else to install — no NeMo,
no other toolkit.

<details>
<summary>Conda environment (recommended for reproducibility)</summary>

```bash
conda create -n indic-transcribe python=3.10 -y
conda activate indic-transcribe
pip install torch torchaudio transformers sentencepiece soundfile
```

</details>

<h3 style="color:#FFD21E;">Input audio requirements</h3>

|  |  |
| --- | --- |
| Sample rate | 16 kHz (resampled automatically if it differs) |
| Channels | Mono |
| Formats | `.wav`, `.flac`, `.mp3` |
| Speakers | Single speaker — see [Limitations](#limitations) |

```bash
# Convert anything to the expected format
ffmpeg -i input.mp3 -ac 1 -ar 16000 -c:a pcm_s16le audio.wav
```

<h3 style="color:#FFD21E;">Basic inference</h3>

```python
# Gated model: accept the terms on this page, then run `hf auth login` once (older installs: `huggingface-cli login`).
import sys
from huggingface_hub import snapshot_download

model_dir = snapshot_download("bodhan-ai/Indic-Transcribe-Flex")
sys.path.insert(0, model_dir)          # the model code ships inside the download

from indic_transcribe import IndicTranscribe
asr = IndicTranscribe.from_pretrained(model_dir)
print(asr("audio.wav", lang="hi"))
```

<h3 style="color:#FFD21E;">The three transcription modes</h3>

The same audio, three outputs. Pick the mode that matches what your downstream system expects.

**Native script**: everything in the language's own script. Recommended for production.

```python
print(asr("audio.wav", lang="hi", mode="native"))
# मैंने कल पांच बजे तीन फाइलें अपलोड कीं
```

**Mixed script (ITN)**: native words in native script; English words and numerals in Latin. Inverse text normalization turns spoken numbers into digits. Recommended if you want formatted and normalized output.

```python
print(asr("audio.wav", lang="hi", mode="mixed"))
# मैंने कल 5 बजे 3 files upload कीं
```

**Romanized**: everything transliterated into Latin script. Useful for search indexing, keyword spotting, and Latin-only UIs.

```python
print(asr("audio.wav", lang="hi", mode="romanized"))
# maine kal 5 baje 3 files upload kin
```

<h3 style="color:#FFD21E;">Automatic language ID + transcription</h3>

When you don't know the language ahead of time, omit `lang`. The model first identifies it, then transcribes at the cost of one decoder step, without requiring a second encoder pass.

```python
text, lid = asr.transcribe("unknown_language.wav", return_lid=True)
print(lid["lang"])   # e.g. "ta"
print(text)
```

A language you supply always wins; identification only fills a gap. `return_lid=True` also
works when you *did* supply one, so a disagreement between your metadata and the model stays
visible instead of silent:

```python
text, lid = asr.transcribe("audio.wav", lang="hi", return_lid=True)
# lid == {"lang": "hi", "source": "explicit", "topk": [("hi", 0.9999), ("ur", 0.0001), ...]}
```

<h3 style="color:#FFD21E;">Language identification only</h3>

To use the model purely as a language-ID system, read the predicted language and discard the transcript.

```python
for path in ["a.wav", "b.wav", "c.wav"]:
    print(path, asr.identify(path))
    # [('ta', 0.9812), ('ml', 0.0104), ('kn', 0.0031), ...]
```

`identify` returns the ranked distribution rather than a single string, because for the
confusable pairs the top-1 alone hides how close the decision was. Accuracy is uneven:
`ml`/`ta` 0.98 and `kn`/`bn` 0.96, against `bho` 0.05, `hi` 0.26, `mai` 0.36 and `ur` 0.49,
each absorbed by a close neighbour. If you have a language label, pass it.



<h3 style="color:#FFD21E;">Long audio</h3>

`asr(...)` decodes a file in one pass and refuses audio longer than 45 s. That limit is
deliberate: the model trains on clips of up to 30 s, and a single pass over long audio
collapses into repetition rather than degrading gracefully. For long audio, use the chunked
helper that ships with the model. It cuts the audio at natural pauses (or at exactly 30 s
when there is none) into pieces of at most 30 s, transcribes each, and joins the text:

```python
from long_form import transcribe_long

print(transcribe_long(asr, "long_audio.wav", lang="hi"))
```

Omit `lang` and the language is identified once, from samples across the whole file.
Pass it whenever you know it — it is faster and avoids a wrong guess.

Output modes work the same way on long audio — pass `mode="mixed"` or `mode="romanized"` to `transcribe_long`.

How the long-audio path works:

```text
audio (.wav / .flac / .mp3, any sample rate, mono or stereo)
  │  load → mono → 16 kHz
  ▼
lang given? ── no ──► identify it (long files: 3 × 20 s windows across the file)
  │
  ▼
≤ 30 s? ── yes ──► decode in one pass ───────────────────────────┐
  │ no                                                           │
  ▼                                                              │
split at pauses                                                  │
  • pause = ≥ 250 ms at least 25 dB below the file's peak        │
  • cut in the middle of the pause nearest 25 s                  │
  • no pause within 30 s → cut at exactly 30 s                   │
  ▼                                                              │
decode each piece (every piece ≤ 30 s)                           │
  • output looping? → split that piece at a pause, decode again  │
  • still looping?  → collapse the repeated phrase               │
  ▼                                                              │
join the pieces ─────────────────────────────────────────────────┴──► transcript
```

Pieces are kept as long as possible because every cut costs a little accuracy — the words
at a cut are the ones most likely to be wrong — and 30 s is the longest input the model
was trained on.

<h3 style="color:#FFD21E;">Command line</h3>

`inference.py` ships with the model and wraps the same calls. It picks the whole-file or
chunked path by duration, and identifies the language when `--lang` is omitted.

```bash
MODEL_DIR=$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("bodhan-ai/Indic-Transcribe-Flex"))')

python "$MODEL_DIR/inference.py" audio.wav --lang hi
python "$MODEL_DIR/inference.py" long_call.wav --show-lang      # no --lang: identify it, and print it
python "$MODEL_DIR/inference.py" *.wav --lang ta                # several files
python "$MODEL_DIR/inference.py" audio.wav --lang hi --mode romanized
```

<h2 style="color:#FFD21E;">NeMo checkpoint</h2>

`nemo/` holds the same model as an NVIDIA NeMo checkpoint, for pipelines already built on NeMo.
Its config names the tokenizer class the model was trained with, which is not part of a stock
NeMo install, so `load_nemo` registers the copy that ships beside it before restoring:

```python
import sys
from huggingface_hub import snapshot_download

model_dir = snapshot_download("bodhan-ai/indic-transcribe-flex")
sys.path.insert(0, f"{model_dir}/nemo")      # the loader ships inside the download

from load_nemo import load_nemo_model
model = load_nemo_model(model_dir)
print(model.transcribe(["audio.wav"], source_lang="hi", target_lang="hi", pnc="yes")[0].text)
```

Nothing inside NeMo is modified, and the loader finds the checkpoint next to itself. If you
would rather not add anything to `sys.path`, copy `nemo/canary_multilingual_tokenizer.py` into
your NeMo install beside `canary_tokenizer.py` and `restore_from` the `.nemo` directly.

**Give it a language.** Pass any of the codes in *Full list of language codes* below as both
`source_lang` and `target_lang`:

```text
as  bho  bn  brx  doi  en  gu  hi  kn  kok  ks  mai  ml  mni
mr  ne  or  pa  sa  sat  sd  ta  te  ur
```

The checkpoint keeps one shared sub-tokenizer for all of them, and the class above routes every
code to it. Without it, stock NeMo raises `RuntimeError: Unsupported language: 'ml'`.

The NeMo path writes each language in its native script. The mixed and romanized modes are
available through the Hugging Face model above (`mode="mixed"` / `mode="romanized"`).

<h3 style="color:#FFD21E;">Command line</h3>

```bash
pip install "nemo_toolkit[asr]"

MODEL_DIR=$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("bodhan-ai/indic-transcribe-flex"))')

python "$MODEL_DIR/nemo/test_nemo.py" audio.wav hi          # check the install
python "$MODEL_DIR/nemo/inference_nemo.py" audio.wav --lang hi
python "$MODEL_DIR/nemo/inference_nemo.py" long_call.wav --lang ta   # chunked automatically
python "$MODEL_DIR/nemo/inference_nemo.py" *.wav --lang bn --batch-size 8
```

<h3 style="color:#FFD21E;">Long audio</h3>

Audio over 30 s is cut at natural pauses into pieces of at most 30 s and decoded piece by
piece -- the same splitter, cap and repetition guard the Hugging Face model uses, so both
paths cut audio identically. `inference_nemo.py` does it automatically; in code:

```python
from long_form_nemo import transcribe_long_nemo

print(transcribe_long_nemo(model, "long_audio.wav", lang="hi", batch_size=8))
```

Batching the pieces makes this path the faster one for long audio: a 10-minute file takes
8 s at `batch_size=8` against 24 s at 1, and an hour of audio runs in 48 s (77x realtime,
9.2 GiB of GPU memory). Pass `lang` -- this path has no language identification of its own,
and a wrong language gives you the wrong script rather than an error.

---

<h2 id="inputs" style="color:#FFD21E;">Input(s)</h2>

| Field | Details |
| :--- | :--- |
| **Input Type(s)** | Audio, Language ID |
| **Input Format(s)** | `.wav`, `.flac`, `.mp3`; string language code |
| **Input Parameters** | One-dimensional (1D) audio; one-dimensional (1D) language ID |
| **Other Properties** | 16 kHz mono; audio is resampled automatically if it differs. Single speaker. |

---

<h2 id="output" style="color:#FFD21E;">Output</h2>

| Field | Details |
| :--- | :--- |
| **Output Type(s)** | Text string in the input language |
| **Output Format(s)** | String |
| **Output Parameters** | One-dimensional (1D) |
| **Other Properties** | Selectable native-script, mixed-script, or romanized rendering; optional detected-language tag. |

---

<h2 id="supported-language-codes" style="color:#FFD21E;">Supported Language Codes</h2>

Pass these to `source_lang`. Use `"auto"` for automatic language identification.

| Example | Value |
| --- | --- |
| Hindi | `source_lang="hi"` |
| Tamil | `source_lang="ta"` |
| Bengali | `source_lang="bn"` |
| Indian English | `source_lang="en"` |
| Auto-detect | `source_lang="auto"` |

All 27 languages use standard ISO 639-1 / 639-3 codes (also listed in the `language:` field at the top of this card).

<details>
<summary>Full list of language codes (27)</summary>

| Language | Code | Script | Group |
| --- | --- | --- | --- |
| English (Indian) | `en` | Latin | Indian-accented English |
| Assamese | `as` | Bengali–Assamese | Scheduled |
| Bengali | `bn` | Bengali | Scheduled |
| Bodo | `brx` | Devanagari | Scheduled |
| Dogri | `doi` | Devanagari | Scheduled |
| Gujarati | `gu` | Gujarati | Scheduled |
| Hindi | `hi` | Devanagari | Scheduled |
| Kannada | `kn` | Kannada | Scheduled |
| Kashmiri | `ks` | Perso-Arabic / Devanagari | Scheduled |
| Konkani | `kok` | Devanagari | Scheduled |
| Maithili | `mai` | Devanagari | Scheduled |
| Malayalam | `ml` | Malayalam | Scheduled |
| Manipuri | `mni` | Bengali / Meetei Mayek | Scheduled |
| Marathi | `mr` | Devanagari | Scheduled |
| Nepali | `ne` | Devanagari | Scheduled |
| Odia | `or` | Odia | Scheduled |
| Punjabi | `pa` | Gurmukhi | Scheduled |
| Sanskrit | `sa` | Devanagari | Scheduled |
| Santali | `sat` | Ol Chiki | Scheduled |
| Sindhi | `sd` | Perso-Arabic / Devanagari | Scheduled |
| Tamil | `ta` | Tamil | Scheduled |
| Telugu | `te` | Telugu | Scheduled |
| Urdu | `ur` | Perso-Arabic | Scheduled |
| Bhojpuri | `bho` | Devanagari | Hindi dialect |
| Chhattisgarhi | `hne` | Devanagari | Hindi dialect |
| Haryanvi | `bgc` | Devanagari | Hindi dialect |
| Bhili | `bhb` | Devanagari | Low resource |

</details>

---

<h2 id="limitations" style="color:#FFD21E;">Limitations</h2>

- **Single-speaker audio.** The model is trained for single-speaker recordings. For multi-speaker scenarios, pair it with a diarization module and transcribe each speaker turn separately.
- **Mixed-script (ITN) mode can be wrong.** Inverse text normalization is genuinely ambiguous in places, so mixed-script mode will occasionally get it wrong. For production, native-script mode is recommended, applying your own ITN downstream if you need it.
- **Romanization is not standardized.** There is no single canonical romanization for most Indian languages. Romanized output follows the model's learned convention, which may not match your existing transliteration scheme.
- **Streaming is not available in `pro`.** Use Indic-Transcribe-lite for real-time use.
- **Low-resource languages are weaker.** Bhili in particular has very little training data. Check the per-language numbers in [Results at a Glance](#results-at-a-glance) before committing to a language.
- **Noise robustness has limits.** Heavily overlapped speech, far-field audio, and very low-SNR recordings will still degrade quality.

---

<h2 id="license--terms-of-use" style="color:#FFD21E;">License / Terms of Use</h2>

Released under [Indic Open Model License v1.0](Bodhan_AI_Open_Model_License.md).

The base model, [nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2), carries its own license terms — ensure your use complies with both.

---

*If you find the license difficult to understand, here is a plain-language guide to the Indic Open Model License.*

Broad, no-cost access for research, government, nonprofit, and commercial use — with a few conditions attached.

> **This deed is a human-readable summary of the license, not a substitute for it.** Where the two disagree, the full **Indic Open Model License** governs.

---

## You're free to

No cost, no royalty, worldwide — for research, government, nonprofit, and commercial use, at any scale.

- ✅ **Run it** — for inference, in a product, in research, however you like.
- ✅ **Change it** — fine-tune, distill, quantize, merge, or otherwise build on it.
- ✅ **Self-host it** — power your own product or service with it, commercial or not.
- ✅ **Share it** — pass on copies of the model or your own version of it.

---

## As long as you

Five conditions cover almost everything. The rest of the license is these, spelled out in legal detail.

### 1. Give credit

Wherever you ship the model or a derivative to anyone else, say where it came from — and don't strip out existing notices.

```
"Built with [Model Name] from Bodhan AI / AI4Bharat."
```

### 2. Pass it on the same way

If you give your fine-tuned or derived version to anyone else — hand it over, or run it as a service for them — it carries this exact license. You can't relicense it on different terms.

### 3. Ask before hosting it for others

Self-hosting is free. But if you're going to run it as an API or hosted service that *other people or companies* call directly, that needs Bodhan AI's written sign-off first — unless you're a nonprofit, government, or academic user, or you publicly release an equally capable open version within 90 days.

### 4. Don't use it to cause harm

No exceptions — not even for nonprofit or research use. That means no:

- child sexual abuse material, or content that sexualizes minors
- weapons development, including chemical, biological, radiological, or nuclear
- mass surveillance or social-scoring systems
- disinformation campaigns, including election manipulation
- automated decisions that affect someone's legal rights without human oversight
- deepfakes or voice clones of real people without their consent
- robocalls, auto-dialers, or voice-phishing scams
- AI companion products designed to simulate romance or foster emotional dependency

### 5. Talk to us if your product gets huge

If your product, built for your own use rather than hosting for others, crosses either threshold, you will need a separate commercial license. This does not apply to nonprofit, government, or academic users.

| Threshold | |
|---|---|
| **500M+** | monthly active users |
| | *or* |
| **$250M+** | annual revenue |

---

## Good to know

- **No warranty.** The model is provided as-is. It isn't tested or certified for safety-critical use — medical, aviation, nuclear, or similar so test thoroughly before relying on it in high-stakes settings.
- **You handle your own compliance.** Export controls, sanctions, and data-protection law (including India's DPDP Act, where it applies) are on you, not Bodhan AI.
- **This deed doesn't replace the license.** It leaves out most of the legal detail — termination, dispute resolution, confidentiality, and more all live in the full text. Read that before you rely on anything here.

---

<h2 id="use-case" style="color:#FFD21E;">Use Case</h2>

Transcription of multilingual and code-mixed Indian-language audio, including native-script, mixed-script, and romanized output, plus language identification.

---

<h2 id="deployment-geography" style="color:#FFD21E;">Deployment Geography</h2>

Global

---

<h2 id="citation" style="color:#FFD21E;">Citation</h2>

```bibtex
@misc{indictranscribe2026,
  title  = {Indic-Transcribe: Built for the way India actually speaks},
  author = {Bodhan AI, AI4Bharat},
  year   = {2026},
  url    = {https://bodhan.ai/research/blogs/indic-transcribe}
}
```

---

<h2 id="ethical-considerations" style="color:#FFD21E;">Ethical Considerations</h2>

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Developers should work with their team to ensure this model meets requirements for the relevant industry and use case, and addresses unforeseen product misuse.