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

# callbacks.py

from __future__ import annotations
from typing import Callable, List

from transformers.integrations import WandbCallback
from transformers import GenerationConfig
import wandb, torch
from tqdm import tqdm


class WandbSampleCallback(WandbCallback):
    """
    Log N sample generations to Weights & Biases after every evaluation step.

    Parameters
    ----------
    trainer : transformers.Trainer
        The trainer instance (needed for model + tokenizer access).
    dataset : datasets.Dataset
        The dataset to draw samples from.
    prompt_builder : Callable[[dict, Any], str]
        Function that turns a dataset example -> chat prompt *string*.
        (We pass StageRecipe.build_prompt from `train_slm.py`.)
    num_samples : int
        How many rows to sample & log.
    max_new_tokens : int
        Generation length for each sample.
    """

    def __init__(
        self,
        cfg,
        trainer,
        dataset,
    ):
        super().__init__()
        self.examples = dataset
        self.model = trainer.model
        self.tok = trainer.tokenizer
        self.gen_cfg = GenerationConfig.from_model_config(
            trainer.model.config
        )  # inherit defaults
        self.gen_cfg.max_new_tokens = cfg.max_new_tokens

    # --------------------------------------------------------------

    @torch.inference_mode()
    def _generate(self, prompt_ids: torch.Tensor) -> str:
        out = self.model.generate(
            prompt_ids,
            generation_config=self.gen_cfg,
        )
        # strip the prompt
        gen = out[0, prompt_ids.size(1) :]
        return self.tok.decode(gen, skip_special_tokens=False)

    # --------------------------------------------------------------

    def _build_table(self) -> wandb.Table:
        cols = ["label", "prompt", "generation"] + list(self.gen_cfg.to_dict().keys())
        table = wandb.Table(columns=cols)

        for ex in tqdm(self.examples, leave=False):
            prompt_txt = ex["eval"]
            prompt_ids = self.tok(prompt_txt, return_tensors="pt").input_ids.to(
                self.model.device
            )
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                gen_txt = self._generate(prompt_ids)

            table.add_data(ex["label"], prompt_txt, gen_txt, *self.gen_cfg.to_dict().values())

        return table

    # --------------------------------------------------------------

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        tbl = self._build_table()
        wandb.log({f"sample_generations/step_{state.global_step}": tbl})

