# Google Drive artifacts manifest

Upload this folder together with `outputs/indic_transcribe_mr.nemo` to the
submission Drive folder. The `.nemo` checkpoint is 4.6 GB and is deliberately
not stored in the Git repository.

| File | Purpose |
|---|---|
| `smoke.log` | 10-step single-GPU end-to-end smoke run |
| `real_run.log` | 600-step two-GPU fine-tuning run |
| `infer.log` | held-out base vs fine-tuned WER evaluation |
| `metrics.csv` | training metrics used for the loss curve |
| `loss_curve.png` | visual training-loss artifact |
| `benchmark.csv` | latency/throughput/memory/WER serving benchmark |
| `benchmark.log` | raw console output for the serving benchmark |
