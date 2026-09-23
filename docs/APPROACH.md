# Approach & Challenges

Fine-tuning `bodhan-ai/indic-transcribe-flex` (ASR) on Marathi, end-to-end.

## 1. Why ASR, and why this model

The brief let me pick MT, ASR, or TTS from the Bodhan family and said metrics are
not graded — the point is a clean, working fine-tuning pipeline. I picked **ASR**
with **`indic-transcribe-flex`** for concrete reasons:

- **Footprint.** It's the smallest of the three Bodhan models (1.2B / 4.6GB) vs
  Indic-Translate (8B) and Indic-Speak (3B). The training box had `/home` at 98%
  full (~70GB free), so the lightest model was the pragmatic call.
- **Tooling.** It's a NeMo model (`EncDecMultiTaskModel`, a Canary-2 derivative:
  FastConformer encoder + Transformer decoder). NeMo has first-class, well-worn
  ASR fine-tuning, which maximises the chance of a genuinely clean end-to-end run.
- **Data availability.** Good, ungated Marathi speech corpora exist (FLEURS,
  Common Voice), and ASR data prep (audio -> manifest) is straightforward.

## 2. Dataset

**Google FLEURS, config `mr_in` (Marathi).** Ungated, already 16 kHz mono (what
the model wants), and small — a few thousand read-speech utterances. I used 800
train / 200 val clips. FLEURS is deliberately modest: the goal is to exercise the
training loop, not chase WER. `src/prepare_data.py` is written so swapping in
Common Voice `mr` for scale is a two-line change (dataset id + config).

Each clip is written to `.wav` and one JSON line is emitted per utterance in NeMo
manifest form, with the fields the Canary multitask stack expects:

```json
{"audio_filepath": "...", "duration": 14.34, "text": "...",
 "lang": "mr", "source_lang": "mr", "target_lang": "mr", "pnc": "yes"}
```

## 3. Fine-tuning strategy

- **Restore the real model the authors' way.** The repo ships a custom tokenizer
  class a stock NeMo install doesn't know; its `nemo/load_nemo.py` registers that
  class before `EncDecMultiTaskModel.restore_from(...)`. I reuse that exact helper
  rather than reimplementing the restore — no guessing about the checkpoint.
- **Freeze the encoder, train the decoder + heads (default).** On a small dataset
  this is the safer choice: it preserves the expensive, well-trained FastConformer
  acoustic encoder (less catastrophic forgetting), trains far fewer parameters, and
  keeps optimizer state small — friendlier to the near-full disk. `--full`
  unfreezes everything for a full fine-tune.
- **AdamW, lr 1e-5, cosine schedule with warmup**, bf16 mixed precision.
- **2× RTX A6000 via DDP** (`find_unused_parameters=True` because the frozen
  encoder produces params with no grad). Lhotse shards data across ranks itself,
  so Lightning's distributed sampler is disabled.

## 4. Challenges (the real ones)

1. **No torchaudio matches this box's torch.** The machine runs a custom
   `torch 2.14.0+cu126`; the installed torchaudio was built for CUDA 12.8, and
   torchaudio's import guard hard-refuses the 12.6-vs-12.8 mismatch. No public
   torchaudio pairs with torch 2.14, so "install the matching one" wasn't an
   option. `src/_env.py` spoofs the reported CUDA string before torchaudio loads
   — purely cosmetic; I verified a MelSpectrogram actually runs on the GPU.
2. **`datasets>=5` wants torchcodec to decode audio.** That would drag in yet
   another torch-pinned dependency. Instead I load FLEURS with `Audio(decode=False)`
   and decode the bytes with `soundfile` myself.
3. **The multilingual tokenizer needs a per-utterance language.** NeMo's Lhotse
   adapter maps a manifest `lang` field to the Lhotse supervision language, which
   the aggregate tokenizer reads. My first manifests only had `source_lang` /
   `target_lang`, so tokenisation asserted; adding `lang` fixed it.
4. **DDP vs Lhotse's dynamic sampler.** Lightning tried to wrap Lhotse's sampler
   (which has no `__len__`) in a `DistributedSampler`. Fixed with
   `use_distributed_sampler=False` — Lhotse handles sharding. This only showed up
   on 2 GPUs; the 1-GPU smoke run passed straight through.

Isolated all of this in a dedicated conda env (`bodhan-asr`) so the machine's
shared `ml` env stayed untouched and the run is reproducible from
`requirements.txt`.

## 5. Results (sanity, not a benchmark)

Smoke run: 10 steps, validation, WER computed, checkpoint saved — the loop closes.
Short run: 2× A6000, 600 steps, encoder frozen.

| model | WER (100 FLEURS `mr` val clips) |
|---|---|
| base `indic-transcribe-flex` | 0.180 |
| fine-tuned (decoder-only, 600 steps) | **0.165** |

train_loss fell 0.67 -> 0.14 over the run (see `outputs/logs/version_1/metrics.csv`).
The qualitative diffs are mostly spelling/spacing normalisation (e.g. नाविन्य ->
नावीन्य, joined vs split compounds).

WER numbers here are illustrative on 200 FLEURS val clips; the base model already
transcribes Marathi well, so this run demonstrates a working fine-tune, not a
metric win — exactly what the brief asked for.

## 6. Reproduce

```bash
conda create -y -n bodhan-asr python=3.10 pip && conda activate bodhan-asr
pip install -r requirements.txt
huggingface-cli login                      # accept the Bodhan license first
python src/prepare_data.py --out data/fleurs_mr
python src/finetune.py --data data/fleurs_mr --smoke              # sanity
python src/finetune.py --data data/fleurs_mr --devices 2 --max-steps 600
python src/infer.py  --data data/fleurs_mr --finetuned outputs/indic_transcribe_mr.nemo
```
