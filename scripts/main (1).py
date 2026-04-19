"""
main.py
-------
Mistral-7B fine-tuning pipeline.

ALL FIXES from the Llama run are applied here:
  FIX 1 — MetricsCallback bug: 'metrics_callback' NameError in save_metrics()
           → callback is now passed as a parameter correctly
  FIX 2 — FP16/BF16 conflict (RuntimeError: _amp_foreach_non_finite_check_and_unscale_cuda)
           → FP16=False, BF16=True in global_variables.py
  FIX 3 — SFTTrainer max_seq_length conflict with newer TRL
           → Using SFTConfig (replaces TrainingArguments for SFTTrainer)
             which uses max_length instead of max_seq_length
  FIX 4 — local_files_only blocked HF Hub download
           → Removed from model loading; model fetches from HF Hub / cache

Run:
    python scripts/main.py                   # train + eval + export
    python scripts/main.py --mode eval       # evaluate saved model
    python scripts/main.py --mode infer      # interactive terminal inference
    python scripts/main.py --skip-merge      # save adapter only, skip merge
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")    # headless — no display needed on terminal
import matplotlib.pyplot as plt

# ── FIX 3: Use SFTConfig instead of TrainingArguments ──
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

# ── project imports ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from global_variables import (
    OUTPUTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, METRICS_DIR, PLOTS_DIR,
    FINETUNED_MODEL_DIR,
    BATCH_SIZE, GRAD_ACCUM_STEPS, NUM_EPOCHS, LEARNING_RATE,
    WARMUP_RATIO, LR_SCHEDULER, WEIGHT_DECAY, FP16, BF16,
    LOGGING_STEPS, EVAL_STEPS, SAVE_STEPS, SAVE_TOTAL_LIMIT,
    REPORT_TO, MAX_SEQ_LENGTH, SEED,
)
from utils import prepare_datasets, load_dataset_auto, validate_records
from models import load_model_for_training, load_finetuned_model, generate_response


# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

for d in (OUTPUTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, METRICS_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "run.log", mode="a"),
    ],
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
#  FIX 1: MetricsCallback — defined at module level, passed explicitly
#  In the Llama run: save_metrics() referenced 'metrics_callback' by name
#  which caused NameError. Now it's passed as a parameter.
# ─────────────────────────────────────────────────────────────────────────────

class MetricsCallback(TrainerCallback):
    """Collect loss and LR at every logging step."""

    def __init__(self):
        self.train_losses:   list = []
        self.eval_losses:    list = []
        self.learning_rates: list = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step
        if "loss" in logs:
            self.train_losses.append((step, logs["loss"]))
        if "eval_loss" in logs:
            self.eval_losses.append((step, logs["eval_loss"]))
        if "learning_rate" in logs:
            self.learning_rates.append((step, logs["learning_rate"]))


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics(callback: MetricsCallback):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Loss curve
    fig, ax = plt.subplots(figsize=(10, 5))
    if callback.train_losses:
        steps, vals = zip(*callback.train_losses)
        ax.plot(steps, vals, label="Train Loss", color="steelblue")
    if callback.eval_losses:
        steps, vals = zip(*callback.eval_losses)
        ax.plot(steps, vals, label="Eval Loss", color="coral", linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Mistral-7B Fine-Tuning: Train & Eval Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = PLOTS_DIR / "loss_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Loss curve saved → {path}")

    # LR schedule
    if callback.learning_rates:
        fig, ax = plt.subplots(figsize=(10, 4))
        steps, lrs = zip(*callback.learning_rates)
        ax.plot(steps, lrs, color="mediumseagreen")
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule (Cosine)")
        ax.grid(True, alpha=0.3)
        path = PLOTS_DIR / "lr_schedule.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"LR schedule saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE METRICS
#  FIX 1 applied: callback passed as explicit parameter, not referenced by name
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics(trainer, callback: MetricsCallback, duration_secs: float):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "total_steps":               trainer.state.global_step,
        "best_metric":               trainer.state.best_metric,
        "train_losses":              callback.train_losses,
        "eval_losses":               callback.eval_losses,
        "learning_rates":            callback.learning_rates,
        "training_duration_seconds": round(duration_secs, 2),
        "training_duration_human":   time.strftime("%H:%M:%S", time.gmtime(duration_secs)),
    }

    path = METRICS_DIR / "training_metrics.json"
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info(f"Metrics saved → {path}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
#  EXPORT MODEL
# ─────────────────────────────────────────────────────────────────────────────

def export_model(model, tokenizer, skip_merge: bool = False):
    FINETUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if skip_merge:
        logger.info(f"Saving LoRA adapter → {FINETUNED_MODEL_DIR}")
        model.save_pretrained(str(FINETUNED_MODEL_DIR))
        tokenizer.save_pretrained(str(FINETUNED_MODEL_DIR))
        logger.info("Adapter saved. Load with PeftModel.from_pretrained().")
    else:
        logger.info("Merging LoRA weights into base model…")
        merged = model.merge_and_unload()
        logger.info(f"Saving merged model → {FINETUNED_MODEL_DIR}")
        merged.save_pretrained(
            str(FINETUNED_MODEL_DIR),
            safe_serialization=True,
            max_shard_size="4GB",
        )
        tokenizer.save_pretrained(str(FINETUNED_MODEL_DIR))
        logger.info("Merged model saved — ready for inference.")


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(skip_merge: bool = False):
    logger.info("=" * 60)
    logger.info("  MISTRAL-7B FINE-TUNING PIPELINE")
    logger.info("=" * 60)

    # ── Step 1: Data ──────────────────────────────────────────────────────────
    logger.info("Step 1/5 — Loading and preparing datasets…")
    train_dataset, eval_dataset = prepare_datasets()
    logger.info(f"  Train samples : {len(train_dataset)}")
    logger.info(f"  Eval  samples : {len(eval_dataset)}")

    # ── Step 2: Model ─────────────────────────────────────────────────────────
    logger.info("Step 2/5 — Loading Mistral-7B and attaching LoRA…")
    model, tokenizer = load_model_for_training()

    # ── Step 3: Trainer config ────────────────────────────────────────────────
    logger.info("Step 3/5 — Configuring SFTTrainer…")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # FIX 3: SFTConfig instead of TrainingArguments
    # SFTConfig uses max_length (not max_seq_length) — avoids the TRL conflict
    sft_config = SFTConfig(
        # Paths
        output_dir=str(CHECKPOINTS_DIR),
        logging_dir=str(LOGS_DIR),

        # Training
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,

        # Optimisation
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        optim="paged_adamw_32bit",

        # FIX 2: BF16=True, FP16=False — matches RTX 4500 Ada native dtype
        fp16=FP16,    # False
        bf16=BF16,    # True

        # Logging & saving
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=REPORT_TO,

        # Misc
        seed=SEED,
        dataloader_num_workers=4,
        group_by_length=True,
        ddp_find_unused_parameters=False,

        # SFT-specific
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,   # ← correct param name for SFTConfig
        packing=False,
    )

    # FIX 1: Create callback BEFORE trainer, pass it explicitly
    callback = MetricsCallback()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        callbacks=[callback],        # ← passed here, no global name needed
    )

    # ── Step 4: Train ─────────────────────────────────────────────────────────
    logger.info("Step 4/5 — Training…")
    t0 = time.time()
    trainer.train()
    duration = time.time() - t0
    logger.info(f"Training complete in {time.strftime('%H:%M:%S', time.gmtime(duration))}")

    # FIX 1: pass callback explicitly — no NameError possible
    save_metrics(trainer, callback, duration)
    plot_metrics(callback)

    # ── Step 5: Export ────────────────────────────────────────────────────────
    logger.info("Step 5/5 — Exporting fine-tuned model…")
    export_model(model, tokenizer, skip_merge=skip_merge)

    logger.info("Pipeline finished successfully.")
    logger.info(f"Fine-tuned model → {FINETUNED_MODEL_DIR}")
    logger.info(f"Outputs          → {OUTPUTS_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
#  EVAL ONLY
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_saved_model():
    logger.info("Loading fine-tuned model for evaluation…")
    model, tokenizer = load_finetuned_model()

    records   = load_dataset_auto()
    records   = validate_records(records)
    split     = int(len(records) * 0.9)
    eval_recs = records[split:][:10]   # cap at 10 for quick check

    logger.info(f"Running inference on {len(eval_recs)} eval samples…")
    results = []
    for i, rec in enumerate(eval_recs):
        prediction = generate_response(model, tokenizer, rec)
        results.append({
            "index":               i,
            "user_query":          rec.get("user_query", ""),
            "expected_thought":    rec.get("thought", ""),
            "expected_tool_calls": rec.get("tool_calls", ""),
            "prediction":          prediction,
        })
        print(f"\n{'─'*60}")
        print(f"[{i+1}] Query      : {rec.get('user_query','')[:120]}")
        print(f"    Expected   : {rec.get('tool_calls','')[:200]}")
        print(f"    Prediction : {prediction[:300]}")

    out_path = METRICS_DIR / "eval_predictions.json"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info(f"Eval predictions saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  INTERACTIVE INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def interactive_inference():
    logger.info("Loading fine-tuned Mistral-7B model…")
    model, tokenizer = load_finetuned_model()
    print("\nInteractive inference — type 'quit' to exit.\n")

    while True:
        user_query = input("User Query : ").strip()
        if user_query.lower() in ("quit", "exit", "q"):
            break
        context = input("Context    (Enter to skip): ").strip()
        record  = {"user_query": user_query, "context": context,
                   "thought": "", "tool_calls": ""}
        print("\nGenerating…")
        response = generate_response(model, tokenizer, record)
        print(f"\nModel Output:\n{response}\n{'─'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Mistral-7B fine-tuning pipeline")
    parser.add_argument(
        "--mode", choices=["train", "eval", "infer"], default="train",
        help="train | eval | infer",
    )
    parser.add_argument(
        "--skip-merge", action="store_true", default=False,
        help="Save LoRA adapter only (no merge into base model)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        train(skip_merge=args.skip_merge)
    elif args.mode == "eval":
        evaluate_saved_model()
    elif args.mode == "infer":
        interactive_inference()


if __name__ == "__main__":
    main()
