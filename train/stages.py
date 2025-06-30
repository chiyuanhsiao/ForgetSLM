# ────────────────────────────────────────────────────────
# stages.py
# ────────────────────────────────────────────────────────
"""Stage-specific prompt builders & dataset loaders."""

from typing import List, Dict, Protocol
from datasets import load_dataset

# === helper ================================================================

def _join_units(units: List[int]) -> str:
    return "".join(f"<|{u}|>" for u in units)


def _to_chat(user: str, assistant: str, tok, add_generation_prompt):
    chat = []
    if user is not None:
        chat.append({"role": "user", "content": user})

    if assistant is not None:
        chat.append({"role": "assistant", "content": assistant})

    return tok.apply_chat_template(
        chat, tokenize=False, add_special_tokens=False, add_generation_prompt=add_generation_prompt,
    )

# === TTS instruction map ===================================================
TTS_INSTRUCTION_MAP: Dict[str, str] = {
    "Please repeat the following words: ": "Please speak out loud the following words: ",
    "Say the following words back to me: ": "Speak out loud the following words back to me: ",
    "Repeat after me: ": "Speak out loud after me: ",
    "I want you to repeat this: ": "I want you to speak out loud this: ",
    "Please repeat the words: ": "Please speak out loud the words: ",
    "Echo the following words: ": "Speak out loud the following words: ",
    "Kindly repeat this phrase: ": "Kindly speak out loud this phrase: ",
    "Say the following words once more: ": "Speak out loud the following words once more: ",
    "Please echo the words: ": "Please speak the following words out loud:",
    "I need you to repeat these words: ": "I need you to speak out loud these words: ",
}

# === Stage recipe protocol ================================================
class StageRecipe(Protocol):
    name: str

    def build_prompt(self, ex, tok, mode): ...

    def load_raw_dataset(self, ds_cfg): ...

# === Actual recipes ========================================================
class LLMRecipe:
    name = "LLM"

    def build_prompt(self, ex, tok, mode):
        if mode == "eval":
            user = ex["input"]
            assistant = None
            return _to_chat(user, assistant, tok, add_generation_prompt=True)
        else:
            user = ex["input"]
            assistant = ex["output"]
            return _to_chat(user, assistant, tok, add_generation_prompt=False)

    def load_raw_dataset(self, ds_cfg):
        return load_dataset(ds_cfg.name, split=ds_cfg.split)

class ASRRecipe:
    name = "ASR"

    def build_prompt(self, ex, tok, mode):
        if mode == "eval":
            user = ex["instruction"] + _join_units(ex["discrete_unit"])
            assistant = None
            return _to_chat(user, assistant, tok, add_generation_prompt=True)
        else:
            user = ex["instruction"] + _join_units(ex["discrete_unit"])
            assistant = ex["transcription"]
            return _to_chat(user, assistant, tok, add_generation_prompt=False)


    def load_raw_dataset(self, ds_cfg):
        return load_dataset(ds_cfg.name, split=ds_cfg.split)


class TTSRecipe:
    name = "TTS"

    def build_prompt(self, ex, tok, mode):
        if mode == "eval":
            user = TTS_INSTRUCTION_MAP[ex["instruction"]] + ex["transcription"]
            assistant = None
            return _to_chat(user, assistant, tok, add_generation_prompt=True) 
        else:
            user = TTS_INSTRUCTION_MAP[ex["instruction"]] + ex["transcription"]
            assistant = ex["interleaf"]
            return _to_chat(user, assistant, tok, add_generation_prompt=False)

    def load_raw_dataset(self, ds_cfg):
        return load_dataset(ds_cfg.name, split=ds_cfg.split)


class SQARecipe:
    name = "SQA"

    def build_prompt(self, ex, tok, mode):
        if mode == "eval":
            user = _join_units(ex["input_unit"])
            assistant = None
            return _to_chat(user, assistant, tok, add_generation_prompt=True) 
        else:
            user = _join_units(ex["input_unit"])
            assistant = (
                ex["input_pseudo"] + "\n\n" + ex["output_pseudo"] + "\n\n" + ex["output_7306_interleaf"]
            )
            return _to_chat(user, assistant, tok, add_generation_prompt=False)

    def load_raw_dataset(self, ds_cfg):
        from data import load_magpie  # local import to avoid circular
        return load_magpie(ds_cfg)


# Public helper -------------------------------------------------------------

def get_recipe(stage: str):
    mapping = {
        "LLM": LLMRecipe(),
        "ASR": ASRRecipe(), 
        "TTS": TTSRecipe(), 
        "SQA": SQARecipe(),
    }
    if stage not in mapping:
        raise ValueError(f"Unknown stage: {stage}")
    return mapping[stage]


