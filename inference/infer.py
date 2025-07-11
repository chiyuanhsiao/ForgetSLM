#!/usr/bin/env python

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

# infer.py

import argparse, os, yaml, gc, tempfile
from pathlib import Path
from typing import List, Dict

import torch, datasets
from datasets import Audio, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import PeftModel
from seamless_communication.models.unit_extractor import UnitExtractor
from seamless_communication.models.vocoder import load_vocoder_model

from utils import set_seed, add_unit_column, build_prompt

# ---------------------------------------------------------------------------

class InferenceRunner:
    def __init__(self, cfg: Dict, rank: int = 0):
        self.cfg = cfg
        self.device = torch.device(f"{cfg['device']}:{rank}" if cfg["device"] == "cuda" else "cpu")
        set_seed(cfg["seed"])
        self._load_model()
        self.unit_extractor = UnitExtractor(
            "xlsr2_1b_v2",
            "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy",
            device=self.device,
            dtype=torch.float16,
        )
        self.vocoder = None
        if cfg["u2s"]["enabled"]:
            self.vocoder = load_vocoder_model(
                cfg["u2s"]["vocoder"], device=self.device, dtype=getattr(torch, cfg["u2s"]["fp"])
            )

    # ------------------------------------------------------------------ #
    def _load_model(self):
        mcfg = self.cfg["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(mcfg["tokenizer"])
        base = AutoModelForCausalLM.from_pretrained(
            mcfg["base_id"],
            torch_dtype=getattr(torch, mcfg["fp"]),
            device_map={"": self.device},
        )
        base.resize_token_embeddings(len(self.tokenizer))
        self.model = PeftModel.from_pretrained(base, mcfg["peft_ckpt"]).eval()

        self.gen_cfg = GenerationConfig(
            max_new_tokens=self.cfg["generation"]["max_new_tokens"],
            do_sample=self.cfg["generation"]["do_sample"],
        )

    # ------------------------------------------------------------------ #
    def _generate_batch(self, unit_batch: List[List[int]]) -> Dict[str, List[str]]:
        prompts = [build_prompt(u, self.tokenizer) for u in unit_batch]
        inp = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(self.device)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outs = self.model.generate(**inp, generation_config=self.gen_cfg)
        decoded = self.tokenizer.batch_decode(
            outs[:, inp.input_ids.shape[-1] :], skip_special_tokens=False
        )
        clean = self.tokenizer.batch_decode(
            outs[:, inp.input_ids.shape[-1] :], skip_special_tokens=True
        )
        return {"interleaf": decoded, "text": clean, "tokens": outs[:, inp.input_ids.shape[-1] :].tolist()}

    # ------------------------------------------------------------------ #
    def _u2s_batch(self, token_batch: List[List[int]]):
        unit_id_start = self.tokenizer.convert_tokens_to_ids("<|0|>")
        unit_id_end = self.tokenizer.convert_tokens_to_ids(f"<|{self.cfg['model']['num_units'] - 1}|>")
        wavs = []
        for toks in token_batch:
            ids = torch.tensor(toks, dtype=torch.long, device=self.device)
            ids = ids[(ids >= unit_id_start) & (ids <= unit_id_end)] - unit_id_start
            if ids.numel() == 0:
                wavs.append({"array": torch.zeros(16000).numpy(), "sampling_rate": 16000})
            else:
                wav = self.vocoder(ids, "eng", -1, dur_prediction=True)
                wavs.append({"array": wav.cpu().detach().numpy().squeeze(), "sampling_rate": 16000})
        return wavs

    # ------------------------------------------------------------------ #
    def run_one_dataset(self, spec: Dict):
        print(f"\n{spec['name']}")

        ds = datasets.load_dataset(spec["name"], split=spec["split"])
        ds = add_unit_column(ds, spec["audio_col"], self.unit_extractor, self.device)
        ds = ds.remove_columns(spec["audio_col"])

        # ---------- Generation -----------------------------------------
        ds = ds.map(
            lambda ex: self._generate_batch(ex["question_unit"]),
            batched=True,
            batch_size=self.cfg["generation"]["batch_size"],
            load_from_cache_file=False,
            num_proc=1,
        )

        # ---------- U2S -------------------------------------------------
        if self.cfg["u2s"]["enabled"]:
            ds = ds.map(
                lambda ex: {"response_speech": self._u2s_batch(ex["tokens"])},
                batched=True,
                batch_size=self.cfg["generation"]["batch_size"],
                load_from_cache_file=False,
                num_proc=1,
            ).cast_column("response_speech", Audio())

        # ---------- Push ------------------------------------------------
        tgt_repo = f"{self.cfg['output_repo_prefix']}_{spec['name'].split('/')[-1]}"
        ds.push_to_hub(tgt_repo, token=self.cfg["push_token"])
        print(f"Pushed to {tgt_repo}")

        # free GPU RAM
        del ds
        gc.collect()
        torch.cuda.empty_cache()

# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    runner = InferenceRunner(cfg, rank=args.gpu)

    for ds_spec in cfg["datasets"]:
        runner.run_one_dataset(ds_spec)


if __name__ == "__main__":
    main()

