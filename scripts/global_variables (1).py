"""
global_variables.py
-------------------
Central config for Mistral-7B fine-tuning.
This file has all the fixes from the Llama run applied:
  - BF16=True, FP16=False  (RTX 4500 Ada is BF16 native, FP16 caused unscale error)
  - local_files_only=False  (model loaded from HuggingFace Hub — the path that worked)
  - SFTConfig used instead of TrainingArguments (avoids max_seq_length conflict)
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
#  ROOT PATHS
# ─────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
DATA_DIR        = PROJECT_ROOT / "data"
SCRIPTS_DIR     = PROJECT_ROOT / "scripts"
OUTPUTS_DIR     = PROJECT_ROOT / "outputs"

CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR        = OUTPUTS_DIR / "logs"
METRICS_DIR     = OUTPUTS_DIR / "metrics"
PLOTS_DIR       = OUTPUTS_DIR / "plots"

# ─────────────────────────────────────────────
#  MODEL — HuggingFace Hub ID
#  !! Do NOT use a local Ollama path — HF Hub is what worked !!
#  Mistral-7B does NOT require a Meta license gate,
#  so no approval wait — it downloads immediately after login.
# ─────────────────────────────────────────────
BASE_MODEL_ID = os.environ.get(
    "BASE_MODEL_ID",
    "mistralai/Mistral-7B-Instruct-v0.3",   # change to v0.1 or v0.2 if preferred
)

# Where the fine-tuned model will be saved after training
FINETUNED_MODEL_DIR = OUTPUTS_DIR / "finetuned_model"

# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────
DATASET_JSON = DATA_DIR / "dataset.json"
DATASET_CSV  = DATA_DIR / "dataset.csv"

# Field names — must match your JSON schema exactly
FIELD_USER_QUERY = "user_query"
FIELD_CONTEXT    = "context"
FIELD_THOUGHT    = "thought"
FIELD_TOOL_CALLS = "tool_calls"

# ─────────────────────────────────────────────
#  TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────
SEED             = 42
MAX_SEQ_LENGTH   = 2048
TRAIN_SPLIT      = 0.90          # 90% train, 10% eval → ~8100 train, ~900 eval
BATCH_SIZE       = 4
GRAD_ACCUM_STEPS = 4             # effective batch size = 4 × 4 = 16
NUM_EPOCHS       = 3
LEARNING_RATE    = 2e-4
WARMUP_RATIO     = 0.05
LR_SCHEDULER     = "cosine"
WEIGHT_DECAY     = 0.01

# ── FIXED from Llama run: RTX 4500 Ada is BF16 native ──
# FP16=True caused: RuntimeError: "_amp_foreach_non_finite_check_and_unscale_cuda"
# not implemented for 'BFloat16'
FP16             = False         # ← MUST be False
BF16             = True          # ← MUST be True for this GPU

# ─────────────────────────────────────────────
#  QLoRA / PEFT
# ─────────────────────────────────────────────
LORA_R           = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05
# Mistral-7B attention + MLP projection layers
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
USE_4BIT         = True
BNB_QUANT_TYPE   = "nf4"
USE_NESTED_QUANT = True

# ─────────────────────────────────────────────
#  LOGGING & SAVING
# ─────────────────────────────────────────────
LOGGING_STEPS    = 25
EVAL_STEPS       = 200
SAVE_STEPS       = 200
SAVE_TOTAL_LIMIT = 3
REPORT_TO        = "none"        # "wandb" | "tensorboard" | "none"

# ─────────────────────────────────────────────
#  INFERENCE / GENERATION
# ─────────────────────────────────────────────
MAX_NEW_TOKENS     = 512
TEMPERATURE        = 0.7
TOP_P              = 0.9
TOP_K              = 50
REPETITION_PENALTY = 1.1
