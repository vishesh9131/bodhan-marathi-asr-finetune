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
