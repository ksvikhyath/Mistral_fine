"""
models.py
---------
Load Mistral-7B from HuggingFace Hub with QLoRA,
attach LoRA adapters, and expose inference.

KEY FIXES vs the Llama run:
  1. local_files_only=False  → downloads from HF Hub (the path that worked)
  2. torch_dtype=torch.bfloat16  → matches BF16=True in global_variables
  3. No FP16/BF16 mismatch — model and trainer use the same dtype
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Union

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
    TaskType,
)

from global_variables import (
    BASE_MODEL_ID,
    FINETUNED_MODEL_DIR,
    USE_4BIT, BNB_QUANT_TYPE, USE_NESTED_QUANT,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES,
    BF16,
    MAX_NEW_TOKENS, TEMPERATURE, TOP_P, TOP_K, REPETITION_PENALTY,
    MAX_SEQ_LENGTH,
)
from utils import format_prompt

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  TOKENIZER
# ─────────────────────────────────────────────────────────────────────────────

def load_tokenizer(model_id: Optional[str] = None) -> AutoTokenizer:
    """Load Mistral tokenizer from HuggingFace Hub."""
    model_id = model_id or BASE_MODEL_ID
    logger.info(f"Loading tokenizer: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=False,
        # local_files_only=False  ← default, downloads if not cached
    )

    # Mistral has no pad token — use eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LENGTH

    return tokenizer


# ─────────────────────────────────────────────────────────────────────────────
#  BASE MODEL  (4-bit QLoRA)
# ─────────────────────────────────────────────────────────────────────────────

def _build_bnb_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantisation config."""
    return BitsAndBytesConfig(
        load_in_4bit=USE_4BIT,
        bnb_4bit_quant_type=BNB_QUANT_TYPE,
        bnb_4bit_use_double_quant=USE_NESTED_QUANT,
        bnb_4bit_compute_dtype=torch.bfloat16,   # BF16 — fixed from Llama run
    )


def load_base_model(model_id: Optional[str] = None) -> AutoModelForCausalLM:
    """
    Load Mistral-7B from HuggingFace Hub with 4-bit quantisation.
    Model is cached locally after first download (~4 GB).
    """
    model_id = model_id or BASE_MODEL_ID
    logger.info(f"Loading base model: {model_id}")
    logger.info("(Will download & cache on first run — ~4 GB for Mistral-7B)")

    bnb_config = _build_bnb_config() if USE_4BIT else None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,              # ← BF16, matches global_variables
        trust_remote_code=False,
        # local_files_only not set → will fetch from Hub / use cache
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    if USE_4BIT:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    logger.info("Base model loaded successfully.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  LoRA
# ─────────────────────────────────────────────────────────────────────────────

def _build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def attach_lora(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    """Wrap base model with trainable LoRA adapters."""
    model = get_peft_model(model, _build_lora_config())

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        f"LoRA attached — trainable: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  COMBINED LOADER  (used by main.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_model_for_training(
    model_id: Optional[str] = None,
) -> tuple:
    """Returns (model_with_lora, tokenizer) ready for SFTTrainer."""
    tokenizer = load_tokenizer(model_id)
    model     = load_base_model(model_id)
    model     = attach_lora(model)
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE LOADER  (loads saved fine-tuned model)
# ─────────────────────────────────────────────────────────────────────────────

def load_finetuned_model(
    finetuned_path: Optional[Union[str, Path]] = None,
    base_model_id:  Optional[str] = None,
) -> tuple:
    """
    Load the saved fine-tuned model for inference.
    Detects automatically whether it's a merged model or a PEFT adapter.
    """
    ft_path  = Path(finetuned_path or FINETUNED_MODEL_DIR)
    base_id  = base_model_id or BASE_MODEL_ID
    bnb_cfg  = _build_bnb_config() if USE_4BIT else None

    # Load tokenizer from finetuned dir if it has one, else from HF Hub
    tok_source = str(ft_path) if (ft_path / "tokenizer_config.json").exists() else base_id
    tokenizer  = load_tokenizer(tok_source)

    is_peft = (ft_path / "adapter_config.json").exists()

    if is_peft:
        logger.info(f"Loading PEFT adapter from {ft_path}")
        base = AutoModelForCausalLM.from_pretrained(
            base_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base, str(ft_path))
    else:
        logger.info(f"Loading merged model from {ft_path}")
        model = AutoModelForCausalLM.from_pretrained(
            str(ft_path),
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def generate_response(
    model:     AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    record:    Dict,
    max_new_tokens:     int   = MAX_NEW_TOKENS,
    temperature:        float = TEMPERATURE,
    top_p:              float = TOP_P,
    top_k:              int   = TOP_K,
    repetition_penalty: float = REPETITION_PENALTY,
) -> str:
    """Run inference on a single record dict, return generated string."""
    prompt = format_prompt(record, include_response=False)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    ).to(model.device)

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=temperature > 0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        output_ids = model.generate(**inputs, generation_config=gen_config)

    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def batch_generate(
    model:     AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    records:   List[Dict],
    **gen_kwargs,
) -> List[str]:
    results = []
    for i, rec in enumerate(records):
        logger.debug(f"Generating {i+1}/{len(records)}")
        results.append(generate_response(model, tokenizer, rec, **gen_kwargs))
    return results
