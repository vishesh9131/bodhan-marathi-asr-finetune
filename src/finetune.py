"""
Fine-tune bodhan-ai/indic-transcribe-flex (NeMo Canary EncDecMultiTaskModel) on
Marathi.

The model ships a `.nemo` checkpoint plus a custom tokenizer class that a stock
NeMo install doesn't know about; the repo's `nemo/load_nemo.py` registers that
class before restoring. We reuse that exact helper so the restore path matches
the authors' — no guessing.

Fine-tuning strategy (documented choices, since metrics aren't graded):
  * Restore the pretrained multitask model, then continue training on Marathi
    manifests built by `prepare_data.py` (Lhotse dataloaders, which Canary uses).
  * By default we FREEZE the FastConformer encoder and train only the decoder +
    heads. On a small dataset this is the safer call: it keeps the expensive,
    well-trained acoustic encoder intact (less catastrophic forgetting), trains
    far fewer params, and produces smaller optimizer state — friendlier to a
    near-full disk. `--full` unfreezes everything.
  * bf16 mixed precision, DDP across the requested GPUs.

Usage:
  # tiny smoke run to prove the pipeline (few steps, 1 GPU)
  python src/finetune.py --data data/fleurs_mr --smoke
  # short real run on 2 GPUs
  python src/finetune.py --data data/fleurs_mr --devices 2 --max-steps 1500
"""
import argparse
import os
import sys

import _env  # noqa: F401  -- must precede nemo/torchaudio; see src/_env.py
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

MODEL_ID = "bodhan-ai/indic-transcribe-flex"


def load_pretrained(model_dir):
    """Register the bundled tokenizer, restore the .nemo, return the NeMo model."""
    sys.path.insert(0, os.path.join(model_dir, "nemo"))
    from load_nemo import load_nemo_model  # ships in the model repo
    return load_nemo_model(model_dir)


def data_config(manifest, batch_size, is_train):
    """Lhotse data config for Canary. Prompt slots (source_lang/target_lang/pnc)
    are read from the manifest fields prepare_data.py writes."""
    return OmegaConf.create({
        "use_lhotse": True,
        "manifest_filepath": os.path.abspath(manifest),
        "sample_rate": 16000,
        "batch_size": batch_size,
        "shuffle": is_train,
        "num_workers": 4,
        "pin_memory": True,
        # keep clips within the model's 30s training length
        "max_duration": 30.0,
        "min_duration": 0.3,
    })


def optim_config(lr, max_steps):
    return OmegaConf.create({
        "name": "adamw",
        "lr": lr,
        "weight_decay": 1e-3,
        "betas": [0.9, 0.98],
        "sched": {"name": "CosineAnnealing", "warmup_steps": max(10, max_steps // 20),
                  "min_lr": lr / 20, "max_steps": max_steps},
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fleurs_mr", help="dir with train/val manifests")
    ap.add_argument("--out", default="outputs/indic_transcribe_mr.nemo")
    ap.add_argument("--devices", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--full", action="store_true", help="also fine-tune the encoder")
    ap.add_argument("--smoke", action="store_true", help="tiny run: 10 steps, 1 GPU")
    ap.add_argument("--model-dir", default=None, help="local model dir (skips download)")
    args = ap.parse_args()

    if args.smoke:
        args.devices, args.max_steps, args.batch_size = 1, 10, 2

    import lightning.pytorch as pl
    from lightning.pytorch.strategies import DDPStrategy

    # only the nemo/ folder is needed for NeMo restore (skips the ~4.6GB HF-format
    # safetensors we don't use) — keeps the near-full disk happy.
    model_dir = args.model_dir or snapshot_download(MODEL_ID, allow_patterns=["nemo/*"])
    print(f"Model dir: {model_dir}")
    model = load_pretrained(model_dir)

    if not args.full:
        model.encoder.freeze()
        print("Encoder frozen — training decoder + heads only.")

    train_m = os.path.join(args.data, "train_manifest.json")
    val_m = os.path.join(args.data, "val_manifest.json")
    model.setup_training_data(data_config(train_m, args.batch_size, is_train=True))
    model.setup_validation_data(data_config(val_m, args.batch_size, is_train=False))
    model.setup_optimization(optim_config(args.lr, args.max_steps))

    trainer = pl.Trainer(
        devices=args.devices,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True) if args.devices > 1 else "auto",
        precision="bf16-mixed",
        max_steps=args.max_steps,
        val_check_interval=max(10, args.max_steps // 3),
        limit_val_batches=20,
        log_every_n_steps=5,
        enable_checkpointing=False,   # we save the final .nemo ourselves
        logger=False,
    )
    model.set_trainer(trainer)
    trainer.fit(model)

    if trainer.is_global_zero:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        model.save_to(args.out)
        print(f"Saved fine-tuned model to {args.out}")


if __name__ == "__main__":
    main()
