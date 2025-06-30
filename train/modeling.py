# ────────────────────────────────────────────────────────
# modeling.py
# ────────────────────────────────────────────────────────
"""Model + LoRA utilities and custom trainer."""

from __future__ import annotations

import copy, os
from typing import Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, PeftModel

# ---------------------------------------------------------------------------
#  main loader ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def load_tokenizer_and_model(cfg, recipe):
    """Load base model + tokenizer, expand vocab, optionally load LoRA."""
    # 0. device ------------------------------------------------------------
    local_rank = os.getenv("LOCAL_RANK")
    device_string = "cuda:" + str(local_rank)
    
    # 1. tokenizer ---------------------------------------------------------
    if cfg.stage in ("ASR"):
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id,
            token=os.environ.get("HF_TOKEN"),  # fallback to env var
        )
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")  # safe default
        unit_toks = [f"<|{i}|>" for i in range(cfg.num_units)]
        tokenizer.add_tokens(unit_toks, special_tokens=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            f"{cfg.load_peft_from}/final_tokenizer",
        )

    # 2.  model ------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": device_string},
        token=os.environ.get("HF_TOKEN"),
    )
    model.resize_token_embeddings(len(tokenizer))

    # 3.  Load PEFT adapter from prev stage or create new LoRA ------------
    if cfg.get("load_peft_from"):
        model = PeftModel.from_pretrained(
            model, f"{cfg.load_peft_from}/final_model", torch_dtype=torch.bfloat16
        )
    else:
        lcfg = cfg.get("lora")
        if lcfg:
            peft_conf = LoraConfig(
                r=lcfg.r,
                lora_alpha=lcfg.alpha,
                lora_dropout=lcfg.dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(lcfg.target_modules),
                modules_to_save=list(lcfg.modules_to_save),
            )
            model = get_peft_model(model, peft_conf)

    return tokenizer, model


# ---------------------------------------------------------------------------
#  snapshot util -------------------------------------------------------------
# ---------------------------------------------------------------------------

def snapshot_state(model) -> Dict[str, torch.Tensor]:
    """Detach + copy cpu weights so we can compute L2 regularisation."""
    return {k: v.clone().cpu() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------------------
#  Custom trainer with L2 reg ------------------------------------------------
# ---------------------------------------------------------------------------

class CLTrainer(SFTTrainer):
    """SFTTrainer + L2 penalty against an old weight snapshot."""

    def __init__(self, *args, old_model: Dict[str, torch.Tensor] | None = None, reg_lambda=1.0, **kw):
        super().__init__(*args, **kw)
        self.old_state_dict = old_model  # cpu tensors
        self.reg_lambda = reg_lambda

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss

        if self.old_state_dict and self.reg_lambda > 0.0:
            l2 = 0.0
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.old_state_dict:
                    old_p = self.old_state_dict[n].to(p.device)
                    l2 += ((p - old_p) ** 2).sum()
            loss = loss + self.reg_lambda * l2

        return (loss, outputs) if return_outputs else loss

