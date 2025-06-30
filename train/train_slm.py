# ────────────────────────────────────────────────────────
# train_slm.py
# ────────────────────────────────────────────────────────
"""Entry-point for one training stage (ASR, TTS or SQA).

Usage
-----
$ python train_slm.py --config configs/sqa.yaml
"""

import argparse, os, random, yaml, torch, json, wandb
from omegaconf import OmegaConf
from trl import SFTConfig, DataCollatorForCompletionOnlyLM

from stages import get_recipe
from data import tokenize_and_format, mix_experience_replay
from modeling import (
    load_tokenizer_and_model,
    snapshot_state,
    CLTrainer,
)
from callbacks import WandbSampleCallback  # updated callback


def main(cfg_path: str):
    # 1.  ── load config ----------------------------------------------------
    cfg = OmegaConf.create(yaml.safe_load(open(cfg_path)))

    # 2.  ── reproducibility ------------------------------------------------
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # 3.  ── stage helpers --------------------------------------------------
    recipe = get_recipe(cfg.stage)
    tokenizer, model = load_tokenizer_and_model(cfg, recipe)

    # 4.  ── dataset --------------------------------------------------------
    raw_ds = recipe.load_raw_dataset(cfg.dataset)
    ds = tokenize_and_format(raw_ds, tokenizer, cfg.dataset.label, recipe.build_prompt, num_proc=cfg.num_proc)
    if cfg.get("replay", {}).get("enabled", False):
        train_ds, eval_ds = mix_experience_replay(ds, tokenizer, cfg)
    else:
        ds = ds.select_columns(["text", "eval", "label"])
        train_ds, eval_ds = ds.select(range(cfg.eval.num_samples, len(ds))), ds.select(range(cfg.eval.num_samples))

    # 5.  ── collator & trainer --------------------------------------------
    collator = DataCollatorForCompletionOnlyLM(
        "<|start_header_id|>assistant<|end_header_id|>", tokenizer=tokenizer
    )

    train_kw = OmegaConf.to_container(cfg.train)
    training_args = SFTConfig(
        output_dir=f"{cfg.model_save_path}/{cfg.run_name}", 
        run_name=cfg.run_name,
        **train_kw,
    )

    old_state = snapshot_state(model)
    trainer = CLTrainer(
        model,
        old_model=old_state,
        reg_lambda=cfg.reg_lambda,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    is_main = trainer.is_world_process_zero()

    # 6.  ── WandB ----------------------------------------------------------
    if is_main:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        trainer.add_callback(
            WandbSampleCallback(
                cfg.eval,
                trainer,
                eval_ds,
            )
        )


    # 7.  ── train ----------------------------------------------------------
    trainer.train()

    # 8.  ── save / finish --------------------------------------------------
    if is_main:
        trainer.save_model(
            f"{cfg.model_save_path}/{cfg.run_name}/final_model"
        )
        tokenizer.save_pretrained(
            f"{cfg.model_save_path}/{cfg.run_name}/final_tokenizer"
        )
        wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)

