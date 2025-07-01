# Analyzing Mitigation Strategies for Catastrophic Forgetting in End-to-End Training of Spoken Language Models  
Chi-Yuan Hsiao · Ke-Han Lu · Kai-Wei Chang · Chih-Kai Yang · Wei-Chih Chen · Hung-yi Lee  
[[arXiv 2505.17496](https://arxiv.org/abs/2505.17496)]

This repository contains **all code, configs, and datasets** used in the paper:

## Abstract
End-to-end training of Spoken Language Models (SLMs) commonly involves adapting pre-trained text-based Large Language Models (LLMs) to the speech modality through multi-stage training on diverse tasks such as ASR, TTS and spoken question answering (SQA). Although this multi-stage continual learning equips LLMs with both speech understanding and generation capabilities, the substantial differences in task and data distributions across stages can lead to catastrophic forgetting, where previously acquired knowledge is lost. This paper investigates catastrophic forgetting and evaluates three mitigation strategies—model merging, discounting the LoRA scaling factor, and experience replay to balance knowledge retention with new learning. Results show that experience replay is the most effective, with further gains achieved by combining it with other methods. These findings provide insights for developing more robust and efficient SLM training pipelines.

## Environment

```bash
git clone https://github.com/chiyuanhsiao/ForgetSLM
cd ForgetSLM

bash build_env.sh         
conda activate ./slm_env
````

`build_env.sh` installs:

* PyTorch 2.3 with GPU support
* HuggingFace Transformers + Datasets + PEFT + TRL
* Seamless-M4T `unit_extractor` / `vocoder_v2`
* MergeKit ≥ 0.9.0
* Weights & Biases client


## Training

### Single-GPU

```bash
cd train

export WANDB_API_KEY=XXXXXXXXXXXXXXXXXXXXXXX        # Weights & Biases
export HF_TOKEN=XXXXXXXXXXXXXXXXXXXXXXXXXXXX        # Hugging Face Hub

python train_slm.py --config configs/sqa.yaml
```

The YAML selects

* base model: **Llama-3.2-11B-Vision-Instruct**
* replay ratios: 5 % each from LLM, ASR, TTS (see `replay:` block)
* mitigation: LoRA (r = 64) + L2 (λ = 1 e-3)

All intermediate checkpoints are saved to `model_ckpt/<RUN_NAME>`.

### Multi-GPU (DDP / Accelerate)

```bash
# DDP via torchrun
cd train
WANDB_API_KEY=XXXXXXXXXXXXXXXXXXXXXXX \
HF_TOKEN=XXXXXXXXXXXXXXXXXXXXXXXXXXXX \
torchrun --standalone --nproc_per_node 4 \
  train_slm.py --config configs/sqa.yaml

# Accelerate
cd train
WANDB_API_KEY=XXXXXXXXXXXXXXXXXXXXXXX \
HF_TOKEN=XXXXXXXXXXXXXXXXXXXXXXXXXXXX \
accelerate launch --num_processes 4 train_slm.py --config configs/sqa.yaml
```

The trainer auto-detects `LOCAL_RANK` and sets
`ddp_find_unused_parameters=False` for efficiency.


## Model Merging (optional)

Model merging proved marginally helpful (§4.5 in the paper).

```bash
pip install mergekit

# e.g. DARE fusion of four training stages
mergekit-yaml model_merging/dare.yml ./merged_ckpt/dare
```

Edit `dare.yml`, `ties.yml`, or `linear.yml` to mix your own checkpoints.


## Inference

```bash
cd inference
python infer.py --cfg configs/sqa_audio.yaml --gpu 0
```

Pipeline:

1. **UnitExtractor** quantises input audio (35 u/s, XLSR-1B codebook-10k).
2. Spoken language model *interleaves* text & speech tokens.
3. (Optional) **Vocoder v2** reconstructs waveform from generated units.
4. Outputs (tokens / text / WAV) are pushed to HuggingFace.

Adjust:

* `datasets:` list to evaluate Spoken Web Questions, Llama-Questions, etc.
* `u2s.enabled: false` if you only need text.
* `generation.batch_size` for VRAM constraints.

## Dataset
The datasets used for training SLMs in multiple stages:
* **ASR / TTS Stage:** [LibriSpeech Interleaving](https://huggingface.co/chiyuanhsiao/datasets?search=ls960_interleaves)
* **SQA Stage:** [Magpie Speech](https://huggingface.co/chiyuanhsiao/datasets?search=magpie_rank)

## Citation

```bibtex
@misc{hsiao2025analyzingmitigationstrategiescatastrophic,
      title={Analyzing Mitigation Strategies for Catastrophic Forgetting in End-to-End Training of Spoken Language Models}, 
      author={Chi-Yuan Hsiao and Ke-Han Lu and Kai-Wei Chang and Chih-Kai Yang and Wei-Chih Chen and Hung-yi Lee},
      year={2025},
      eprint={2505.17496},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.17496}, 
}
```


> **Questions / Pull-requests are welcome!**
> For research collaborations please contact the first author (Chi-Yuan Hsiao) at *r12942086@ntu.edu.tw*.



