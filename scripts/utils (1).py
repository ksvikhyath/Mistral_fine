"""
utils.py
--------
Data loading, validation, and prompt formatting for Mistral-7B.

Mistral uses a different chat template from Llama-2:
  <s>[INST] ... [/INST] ... </s>
  (no <<SYS>> block — system prompt goes inside [INST])
"""

import json
import csv
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from datasets import Dataset

from global_variables import (
    DATASET_JSON, DATASET_CSV,
    FIELD_USER_QUERY, FIELD_CONTEXT, FIELD_THOUGHT, FIELD_TOOL_CALLS,
    TRAIN_SPLIT, SEED,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  MISTRAL PROMPT TEMPLATE
#  Mistral-Instruct format — no <<SYS>> block, system goes inside [INST]
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an intelligent AI assistant that plans and executes complex tasks using tools. "
    "Given a user query and context, you must:\n"
    "1. Understand the full requirement\n"
    "2. Think step-by-step to decide what to do\n"
    "3. Output the correct tool call(s) in the required format\n"
    "4. Ensure all user requirements are covered"
)

def format_prompt(record: Dict, include_response: bool = True) -> str:
    """
    Format one dataset record into the Mistral-Instruct chat template.

    Training  → include_response=True  (model sees input + expected output)
    Inference → include_response=False (model generates the output)

    Mistral template:
        <s>[INST] {system}\\n\\n{user_turn} [/INST] {response} </s>
    """
    user_query = record.get(FIELD_USER_QUERY, "").strip()
    context    = record.get(FIELD_CONTEXT,    "").strip()
    thought    = record.get(FIELD_THOUGHT,    "").strip()
    tool_calls = record.get(FIELD_TOOL_CALLS, "").strip()

    # Build user section
    user_section = f"### User Query:\n{user_query}"
    if context:
        user_section += f"\n\n### Context:\n{context}"

    # Full prompt
    prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_section} [/INST]"

    if include_response:
        response_parts = []
        if thought:
            response_parts.append(f"### Thought:\n{thought}")
        if tool_calls:
            response_parts.append(f"### Tool Calls:\n{tool_calls}")
        response = "\n\n".join(response_parts)
        prompt += f" {response} </s>"

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
#  LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_json_dataset(path: Optional[Path] = None) -> List[Dict]:
    path = Path(path or DATASET_JSON)
    if not path.exists():
        raise FileNotFoundError(f"JSON dataset not found: {path}")

    logger.info(f"Loading JSON dataset from {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Handle wrapped formats: {"data": [...]}
    if isinstance(data, dict):
        for key in ("data", "records", "examples", "samples"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError("JSON dataset must be a list of records.")

    logger.info(f"Loaded {len(data)} records from JSON.")
    return data


def load_csv_dataset(path: Optional[Path] = None) -> List[Dict]:
    path = Path(path or DATASET_CSV)
    if not path.exists():
        raise FileNotFoundError(f"CSV dataset not found: {path}")

    logger.info(f"Loading CSV dataset from {path}")
    records = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append(dict(row))

    logger.info(f"Loaded {len(records)} records from CSV.")
    return records


def load_dataset_auto(json_path: Optional[Path] = None,
                      csv_path:  Optional[Path] = None) -> List[Dict]:
    """Try JSON first, then CSV."""
    json_p = Path(json_path or DATASET_JSON)
    csv_p  = Path(csv_path  or DATASET_CSV)

    if json_p.exists():
        return load_json_dataset(json_p)
    if csv_p.exists():
        return load_csv_dataset(csv_p)

    raise FileNotFoundError(
        f"No dataset found.\n  Looked for: {json_p}\n  and: {csv_p}\n"
        "Place your file in the data/ directory."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_FIELDS = {FIELD_USER_QUERY, FIELD_TOOL_CALLS}

def validate_records(records: List[Dict]) -> List[Dict]:
    clean, dropped = [], 0
    for i, rec in enumerate(records):
        missing = REQUIRED_FIELDS - rec.keys()
        if missing:
            logger.warning(f"Record {i} missing required fields {missing} — skipped.")
            dropped += 1
            continue
        for f in REQUIRED_FIELDS:
            if not str(rec.get(f, "")).strip():
                logger.warning(f"Record {i} has empty '{f}'.")
        clean.append(rec)

    if dropped:
        logger.warning(f"Dropped {dropped} invalid records. Remaining: {len(clean)}")
    else:
        logger.info(f"All {len(clean)} records passed validation.")
    return clean


# ─────────────────────────────────────────────────────────────────────────────
#  HuggingFace DATASET CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def records_to_hf_dataset(records: List[Dict]) -> Dataset:
    """Convert list of dicts → HuggingFace Dataset with a 'text' column."""
    rows = {"text": [format_prompt(r, include_response=True) for r in records]}
    return Dataset.from_dict(rows)


def split_dataset(records: List[Dict],
                  train_ratio: float = TRAIN_SPLIT,
                  seed: int = SEED) -> Tuple[Dataset, Dataset]:
    """Shuffle → split → return (train_dataset, eval_dataset)."""
    random.seed(seed)
    shuffled = records[:]
    random.shuffle(shuffled)

    split_idx  = int(len(shuffled) * train_ratio)
    train_recs = shuffled[:split_idx]
    eval_recs  = shuffled[split_idx:]

    logger.info(f"Split → train: {len(train_recs)}, eval: {len(eval_recs)}")
    return records_to_hf_dataset(train_recs), records_to_hf_dataset(eval_recs)


def prepare_datasets(json_path: Optional[Path] = None,
                     csv_path:  Optional[Path] = None) -> Tuple[Dataset, Dataset]:
    """Full pipeline: load → validate → format → split."""
    records = load_dataset_auto(json_path, csv_path)
    records = validate_records(records)
    return split_dataset(records)
