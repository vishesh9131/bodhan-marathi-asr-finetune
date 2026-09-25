# Fine-tuning Bodhan Indic-Transcribe (ASR) on Marathi

Take-home for the AI Research Engineer role (AI4Bharat / Bodhan AI). Fine-tunes
[`bodhan-ai/indic-transcribe-flex`](https://huggingface.co/bodhan-ai/indic-transcribe-flex)
— a 1.2B NeMo ASR model (FastConformer encoder + Transformer decoder, a
`nvidia/canary-1b-v2` derivative) — on Marathi speech, end-to-end on 2× A6000.

The point of the exercise is a clean, working fine-tuning pipeline (metrics are
not graded). Full reasoning and the bugs hit along the way are in
[`docs/APPROACH.md`](docs/APPROACH.md); [`PROGRESS.md`](PROGRESS.md) is the
blow-by-blow log.

## Result
| model | WER (100 FLEURS `mr` val clips) |
|---|---|
| base `indic-transcribe-flex` | 0.180 |
| fine-tuned (decoder-only, 600 steps) | 0.165 |

train_loss 0.67 -> 0.14 over 600 steps:

![loss curve](artifacts/loss_curve.png)

## Serving benchmark

On 100 held-out Marathi clips (1,315.62 seconds of audio) on one RTX A6000, the
fine-tuned model holds **0.1648 WER** from batch 1 through batch 8 while
throughput rises from **15.191** to **44.444 audio seconds/second**. Batch 4 is
the recommended balanced configuration: 39.507 audio seconds/second with 1.878
seconds p95 request latency. Full methodology and the complete table are in
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md); raw results are in
[`artifacts/benchmark.csv`](artifacts/benchmark.csv).

## Layout
- `src/prepare_data.py` — Marathi dataset (Google FLEURS `mr_in`) -> NeMo manifests.
- `src/finetune.py` — restore the model, freeze encoder, train decoder on 2× A6000 (DDP, bf16).
- `src/infer.py` — transcribe + WER, base vs fine-tuned.
- `src/benchmark.py` — reproducible latency, throughput, memory, and WER study
  across inference batch sizes.
- `src/_env.py` — torch/torchaudio CUDA-mismatch shim (see APPROACH §4).
- `artifacts/` — loss curve, metrics.csv, run logs.
- `docs/DEPLOYMENT.md` — benchmark method and deployment recommendation.
- `docs/SUBMISSION.md` — exact GitHub/Drive handoff contents and email template.
- `brief/` — the original assignment PDF and email.

## Quickstart
```bash
conda create -y -n bodhan-asr python=3.10 pip && conda activate bodhan-asr
pip install -r requirements.txt
huggingface-cli login                      # accept the Bodhan license on the model page first
python src/prepare_data.py --out data/fleurs_mr
python src/finetune.py --data data/fleurs_mr --smoke               # 10-step sanity run
python src/finetune.py --data data/fleurs_mr --devices 2 --max-steps 600
python src/infer.py  --data data/fleurs_mr --finetuned outputs/indic_transcribe_mr.nemo
CUDA_VISIBLE_DEVICES=1 python src/benchmark.py --data data/fleurs_mr \
  --finetuned outputs/indic_transcribe_mr.nemo --limit 100
```

The fine-tuned checkpoint (`outputs/indic_transcribe_mr.nemo`, 4.6GB) is shared
via Google Drive rather than git.
