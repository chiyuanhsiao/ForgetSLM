#   Copyright 2025 Chi-Yuan Hsiao (蕭淇元)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# utils.py

from typing import Any
import random, torch, numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from seamless_communication.models.unit_extractor import UnitExtractor


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------- dataset helpers ------------------------------------------------
def add_unit_column(
    ds: Dataset,
    audio_col: str,
    extractor: UnitExtractor,
    device: torch.device,
    units_per_s: int = 35,
) -> Dataset:
    def _fn(ex):
        wav = torch.tensor(ex[audio_col]["array"]).to(device)
        ex["question_unit"] = extractor.predict(wav, units_per_s - 1).tolist()
        return ex

    return ds.map(_fn, num_proc=1, load_from_cache_file=False)


def build_prompt(unit_ids: list[int], tok: AutoTokenizer) -> str:
    unit_seq = "".join(f"<|{u}|>" for u in unit_ids)
    chat = [{"role": "user", "content": [{"type": "text", "text": unit_seq}]}]
    return tok.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )

