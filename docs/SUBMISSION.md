# Submission package

This repository is the GitHub deliverable for the AI Research Engineer take-home
assignment. The large binary checkpoint and execution artifacts belong in the
Google Drive deliverable rather than in Git.

## GitHub repository

`https://github.com/vishesh9131/bodhan-marathi-asr-finetune`

The repository contains the reproducible source code, the approach and
challenge notes, training/inference artifacts small enough for Git, and the
inference-serving benchmark.

## Google Drive upload contents

Create one folder named `bodhan-marathi-asr-finetune-deliverables` and upload:

```text
indic_transcribe_mr.nemo       # outputs/indic_transcribe_mr.nemo (4.6 GB)
artifacts/
  smoke.log
  real_run.log
  infer.log
  metrics.csv
  loss_curve.png
  benchmark.csv
  benchmark.log
```

After uploading, set the folder to **Anyone with the link: Viewer** (or share it
directly with the reviewers) and paste the folder URL in the email reply. The
checkpoint is intentionally excluded from Git by `.gitignore` because it is
4.6 GB.

## Documentation to point reviewers to

- `README.md` — project overview, result, and run commands.
- `docs/APPROACH.md` — model/data/strategy choices and four concrete issues
  diagnosed during the run.
- `docs/DEPLOYMENT.md` — inference benchmark method, results, and serving
  decision.
- `PROGRESS.md` — chronological engineering log.

## Email template

> Hi,
>
> Please find my completed take-home assignment for the AI Research Engineer
> role. I fine-tuned Bodhan's `indic-transcribe-flex` ASR model on Marathi using
> FLEURS, with a decoder-only fine-tuning run on 2× RTX A6000 GPUs. The held-out
> WER improved from 0.180 to 0.165. I also included an inference deployment
> benchmark covering latency, throughput, memory, and batch-size trade-offs.
>
> - GitHub repository: https://github.com/vishesh9131/bodhan-marathi-asr-finetune
> - Google Drive artifacts: <paste shared Drive folder link>
> - Detailed approach and challenges: `docs/APPROACH.md` in the repository
> - Deployment benchmark: `docs/DEPLOYMENT.md` in the repository
>
> Thank you for your consideration.

