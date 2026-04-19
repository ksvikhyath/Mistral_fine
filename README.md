# Mistral-7B Fine-Tuning Pipeline


## How to get Mistral-7B onto your system

Mistral-7B does **not** require a license approval gate (unlike Meta's Llama models).
You just need a free HuggingFace account.

### Step 1 — Get a HuggingFace token

1. Go to https://huggingface.co/settings/tokens
2. Click **New token** → give it a name → select **Read** role → copy it

### Step 2 — Login on your machine

```bash
pip install huggingface_hub
huggingface-cli login
# paste your token when prompted
```

### Step 3 — Download the model

The model downloads automatically on your first `python scripts/main.py` run.
It caches to `~/.cache/huggingface/hub/` (~14 GB for full weights, ~4 GB loaded in 4-bit).

If you want to pre-download manually (e.g. to avoid interruption during training):

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='mistralai/Mistral-7B-Instruct-v0.3',
    local_dir='/home/$(whoami)/models/mistral-7b'
)
"
```

Then update `global_variables.py`:
```python
BASE_MODEL_ID = "/home/yourname/models/mistral-7b"  # local path
```

### If your machine has no internet (university/lab network)

Download on any machine with internet, then transfer via SCP:

```bash
# On internet machine
python -c "
from huggingface_hub import snapshot_download
snapshot_download('mistralai/Mistral-7B-Instruct-v0.3', local_dir='./mistral-7b')
"
scp -r ./mistral-7b yourname@labmachine:/home/yourname/models/

# Then on lab machine, set in global_variables.py:
# BASE_MODEL_ID = "/home/yourname/models/mistral-7b"
```

---

## Project Structure

```
mistral_finetune/
├── data/
│   └── dataset.json          ← your 9k records
├── scripts/
│   ├── global_variables.py   ← all config (edit this first)
│   ├── utils.py              ← data loading + Mistral prompt formatting
│   ├── models.py             ← model loading, LoRA, inference
│   └── main.py               ← training / eval / infer entry point
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   ├── metrics/
│   ├── plots/
│   └── finetuned_model/
└── requirements.txt
```

---

## Setup

```bash
# 1. Create virtual environment
python3 -m venv loravenv
source loravenv/bin/activate

# 2. Install PyTorch with CUDA 12.1 (RTX 4500 Ada)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Login to HuggingFace
huggingface-cli login

# 5. Add your dataset
cp /path/to/your/dataset.json data/dataset.json
```

---

## Running

```bash
# Full training + export
python scripts/main.py

# Save adapter only (faster, smaller)
python scripts/main.py --skip-merge

# Evaluate saved model on eval split
python scripts/main.py --mode eval

# Interactive inference in terminal
python scripts/main.py --mode infer
```

---

## Dataset format

```json
[
  {
    "user_query":  "Book a flight from NYC to Paris",
    "context":     "Budget $1200 per person",
    "thought":     "I need to search for available flights...",
    "tool_calls":  "search_flights(origin='JFK', destination='CDG'...)"
  }
]
```

---

## Key config values (global_variables.py)

| Setting | Value | Why |
|---|---|---|
| `BF16` | `True` | RTX 4500 Ada is BF16-native |
| `FP16` | `False` | FP16 causes unscale error on this GPU |
| `LORA_R` | 16 | Good rank for 9k dataset |
| `BATCH_SIZE` | 4 | Safe for 24 GB VRAM |
| `GRAD_ACCUM_STEPS` | 4 | Effective batch = 16 |
| `NUM_EPOCHS` | 3 | Standard for this dataset size |
| `MAX_SEQ_LENGTH` | 2048 | Mistral context window |

---

## Expected training time

With 9k samples, 3 epochs, batch=4, grad_accum=4 on RTX 4500 Ada:
~3.5 to 4.5 hours (similar to your Llama run which completed in 3:34:22)
