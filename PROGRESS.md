# Progress Log — Fine-tuning Bodhan Indic-Transcribe (ASR) on Marathi

Running log of the assignment. Newest phase appended at the bottom. Each entry:
what I'm about to do, why, then the result. Kept terse on purpose.

## Assignment (from `brief/`)
- Fine-tune ONE Bodhan AI model (MT / ASR / TTS) on Marathi/Bhili, end-to-end.
- Not scored on metrics — scored on a working pipeline, code quality, judgement.
- Deliverables: GitHub repo, Google Drive artifacts, approach docs.
- Deadline: 24 Sep 2026, 6 PM.

## Decisions
- **Task: ASR.** Model: `bodhan-ai/indic-transcribe-flex` (1.2B, NeMo, based on
  `nvidia/canary-1b-v2`: FastConformer encoder + Transformer decoder, 4.6GB ckpt).
  Chosen over MT (8B, big download) and TTS (3B, fiddliest) because it is the
  smallest footprint — matters, `/home` is 98% full (~73GB free) — and NeMo has
  first-class ASR fine-tuning, so the end-to-end run is most likely to be clean.
- **Dataset:** Marathi (`mr`) subset of Mozilla Common Voice (ungated). Audio is
  resampled to 16kHz mono as the model expects. Small subset to fit disk + quota.
- **Compute:** 2× RTX A6000 (of 4), as requested. DDP via NeMo/Lightning.
- **Env:** dedicated conda env `bodhan-asr` so the shared `ml` env is untouched
  and the run is reproducible from `requirements.txt`.
- **Repo:** private GitHub under `vishesh9131`.

## Plan
- P0 Setup: env + NeMo, HF auth, download model + dataset subset.
- P1 Data: Common Voice `mr` -> 16kHz mono wav -> NeMo manifests (train/val).
- P2 Fine-tune: smoke run (few steps) then a short real run on 2 GPUs; checkpoint.
- P3 Inference: transcribe held-out clips; WER before vs after (illustrative only).
- P4 Deliverables: README + approach doc, push repo, artifacts to Drive.

## Blockers / needs from user
- HF token that has accepted the Bodhan Open Model License (model is `gated: auto`).
- Google Drive upload at the end needs the user's auth (manual step).

---

## P0 — Setup

### Building env `bodhan-asr` (background)
About to: create conda env `bodhan-asr` (py3.10) and install `nemo_toolkit[asr]`
+ huggingface_hub, librosa, jiwer, soundfile, datasets. Runs in background; log at
`outputs/env_build.log`. Reason for a separate env: NeMo pins its own torch and I
won't risk downgrading torch in the shared `ml` env.

### Done in P0 so far
- Repo scaffolded, private, pushed: github.com/vishesh9131/bodhan-marathi-asr-finetune
- `src/prepare_data.py` written: FLEURS `mr_in` -> 16kHz wav + NeMo multitask manifests
  (fields: audio_filepath, duration, text, source_lang=target_lang=mr, pnc=yes).
- Env `bodhan-asr` build running in background (NeMo + hf_hub + librosa + jiwer).

### Waiting on
- Env build to finish -> then run prepare_data.py + download model.
- User to accept the Bodhan license + `huggingface-cli login` (model is gated:auto).
  Needed both to download weights AND to read the model's shipped scripts
  (`nemo/inference_nemo.py`, `indic_transcribe.py`) that define the exact
  restore/fine-tune API — so `finetune.py` is deliberately not written yet
  (won't guess the API and risk a wrong pipeline).

### Model API confirmed (read the shipped scripts with the token)
- Class: `nemo.collections.asr.models.EncDecMultiTaskModel` (Canary multitask).
- Restore: repo's `nemo/load_nemo.py` registers a custom `CanaryMultilingualTokenizer`
  then `EncDecMultiTaskModel.restore_from(nemo/indic_transcribe_flex.nemo)`.
- Inference: `model.transcribe(files, source_lang, target_lang, pnc="yes")`.
- Arch: FastConformer enc (32L) + Transformer dec (24L), d_model 1024, 128 mel bins,
  8x subsampling, vocab 7152, trains on <=30s clips. README licenses fine-tuning
  but ships no recipe.

### Wrote finetune.py + infer.py against that real API
- `finetune.py`: reuse `load_nemo_model`, freeze encoder by default (train decoder
  only — safer on small data, lighter on disk), Lhotse manifests, bf16, DDP over
  `--devices`, `--smoke` for a 10-step 1-GPU sanity run. Saves final `.nemo`.
- `infer.py`: base-vs-finetuned WER on val + 3 qualitative examples (jiwer).
- Download trimmed to `allow_patterns=["nemo/*"]` (~4.6GB, skips HF safetensors).

### Now waiting on the env build (slow network) to run: prepare_data -> smoke -> real run.

## P0 result — env works, downloads running
- Env `bodhan-asr`: NeMo 3.0.0, lhotse 1.33, jiwer 4.0, torch 2.14.0+cu126 (4 GPUs).
- Snag: torch here is a custom 2.14+cu126 build in ~/.local; the paired torchaudio
  was built for CUDA 12.8, so torchaudio's import guard raises (12.6 vs 12.8).
  Same break exists in the `ml` env — it's global, not mine. No public torchaudio
  matches torch 2.14. Fix: `src/_env.py` spoofs the reported CUDA string before
  torchaudio loads (cosmetic; verified MelSpectrogram runs on GPU, NeMo imports).
- Downloads running in background: model `nemo/` (~4.6GB, ~370kB/s, ~3h) +
  FLEURS `mr_in`. Smoke run is gated on the model finishing.
- Disk holding at ~72G free.
