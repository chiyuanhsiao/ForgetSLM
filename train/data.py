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

# ────────────────────────────────────────────────────────
# data.py
# ────────────────────────────────────────────────────────
"""Dataset helpers."""

import os
from typing import Callable, List, Dict
from datasets import load_dataset, concatenate_datasets, Dataset
import random

from stages import get_recipe                     # our StageRecipe factory


# --------------------------------------------------------
#  Magpie loader (for SQA)
# --------------------------------------------------------

def load_magpie(cfg) -> Dataset:
    """Concatenate all requested rank/chunk splits."""
    ranks = cfg.get("ranks", [0])
    # support 0-9 style ranges
    if isinstance(cfg.chunks, str) and "-" in cfg.chunks:
        start, end = map(int, cfg.chunks.split("-"))
        chunks = list(range(start, end + 1))
    else:
        chunks = cfg.get("chunks", [0])

    ds = None
    for r in ranks:
        for c in chunks:
            split_name = cfg.name.format(rank=r, chunk=c)
            part = load_dataset(split_name, split=cfg.split)
            part = part.remove_columns(
                [
                    "output_speech_cmu-arctic-xvectors_7306",
                    "input_speech",
                    "output_speech",
                ]
            )
            ds = part if ds is None else concatenate_datasets([ds, part])

    # category filter -------------------------------------------------------
    bad = set(cfg.get("filter_categories", []))
    if bad:
        ds = ds.filter(lambda ex: ex["task_category"] not in bad)

    return ds


# --------------------------------------------------------
#  tokenisation & prompt formatting
# --------------------------------------------------------

def tokenize_and_format(ds: Dataset, tok, label, build_prompt: Callable, num_proc: int | None = None):
    """Add a `.text` column containing the rendered chat prompt."""
    def _fmt(ex):
        ex["text"] = build_prompt(ex, tok, mode="train")
        ex["eval"] = build_prompt(ex, tok, mode="eval")
        ex["label"] = ex[label]
        return ex

    return ds.map(_fmt, num_proc=num_proc)


# ---------------------------------------------------------------------
#  Experience Replay
# ---------------------------------------------------------------------


def mix_experience_replay(
    current_ds: Dataset,
    tokenizer,
    cfg, 
) -> Dataset:
    """
    Blend the *current* dataset with examples from previous tasks
    according to the YAML `replay` block.

    Parameters
    ----------
    current_ds : Dataset
        The freshly tokenised dataset for the task we are training on now.
    replay_cfg : Dict
        A mapping produced by OmegaConf/YAML, e.g.
        {
          enabled: True,
          olds: [
            {stage: "LLM", ratio: 0.05, dataset: {...}},
            {stage: "ASR", ratio: 0.05, dataset: {...}},
            ...
          ]
        }
    seed : int
        RNG seed for reproducible sampling.

    Returns
    -------
    Dataset
        A shuffled dataset containing `current_ds` + replay samples.
    """
    if not cfg.get("replay", {}).get("enabled", False):
        return current_ds
    else:
        replay_cfg = cfg.replay

    random.seed(cfg.seed)
    size_curr = len(current_ds)
    current_ds = current_ds.select_columns(["text", "eval", "label"])
    train_current, eval_current = current_ds.select(range(cfg.eval.num_samples, len(current_ds))), current_ds.select(range(cfg.eval.num_samples)) 
    train_blended: List[Dataset] = [train_current]   # first element: current task
    eval_blended: List[Dataset] = [eval_current]   # first element: current task

    for block in replay_cfg.get("olds", []):
        ratio = float(block.get("ratio", 0.0))
        if ratio <= 0.0:
            continue  # skip if nothing requested

        # ------------------------------------------------------------------
        # 1. Load raw replay data through the StageRecipe so prompt-building
        #    stays consistent with that stage.
        # ------------------------------------------------------------------
        stage_name = block["stage"]
        recipe     = get_recipe(stage_name)
        raw_ds     = recipe.load_raw_dataset(block.dataset)

        # ------------------------------------------------------------------
        # 2. Pick exactly ⌊size_curr · ratio⌋ rows.
        #    • sample *with replacement* if the replay dataset is too small
        #      (common for low-resource tasks).
        # ------------------------------------------------------------------
        quota = int(size_curr * ratio)
        if quota == 0:
            continue

        if len(raw_ds) >= quota:
            idx = random.sample(range(len(raw_ds)), quota)
        else:  # sample with replacement
            idx = [random.randrange(len(raw_ds)) for _ in range(quota)]

        subset = raw_ds.select(idx)

        # ------------------------------------------------------------------
        # 3. Apply that stage’s prompt builder so every row already has
        #    the `"text"` field your Trainer expects.
        # ------------------------------------------------------------------
        subset = tokenize_and_format(subset, tokenizer, block.dataset.label, recipe.build_prompt, num_proc=cfg.num_proc)
        subset = subset.select_columns(["text", "eval", "label"])
        train_subset, eval_subset = subset.select(range(cfg.eval.num_samples, len(subset))), subset.select(range(cfg.eval.num_samples)) 

        train_blended.append(train_subset)
        eval_blended.append(eval_subset)

    # ----------------------------------------------------------------------
    # 4. Concatenate + final global shuffle
    # ----------------------------------------------------------------------
    train_ds = concatenate_datasets(train_blended).shuffle(seed=cfg.seed)
    eval_ds = concatenate_datasets(eval_blended).shuffle(seed=cfg.seed)
    return train_ds, eval_ds

