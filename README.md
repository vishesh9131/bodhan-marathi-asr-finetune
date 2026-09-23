# Fine-tuning Bodhan Indic-Transcribe (ASR) on Marathi

Take-home for the AI Research Engineer role (AI4Bharat / Bodhan AI). Fine-tunes
[`bodhan-ai/indic-transcribe-flex`](https://huggingface.co/bodhan-ai/indic-transcribe-flex)
— a 1.2B NeMo ASR model (FastConformer encoder + Transformer decoder, based on
`nvidia/canary-1b-v2`) — on Marathi speech, end-to-end.

Status: work in progress. See [`PROGRESS.md`](PROGRESS.md) for the running log and
the full write-up of choices and challenges.

## Layout
- `src/prepare_data.py` — Marathi dataset (Google FLEURS `mr_in`) -> NeMo manifests.
- `src/finetune.py` — fine-tune on 2× A6000 (added once the model API is confirmed).
- `src/infer.py` — transcribe + WER before/after.
- `configs/` — training config.
- `brief/` — the original assignment PDF and email.

## Quickstart
```bash
conda create -y -n bodhan-asr python=3.10 pip && conda activate bodhan-asr
pip install -r requirements.txt
huggingface-cli login          # accept the Bodhan license on the model page first
python src/prepare_data.py --out data/fleurs_mr
```
