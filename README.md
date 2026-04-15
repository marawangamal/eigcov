# EigenCov

Data-free model merging via eigenvalue covariance estimation of task vectors.


## Setup

> **Note:** Vision/language and OLMo environments conflict, so separate venvs are used.

```sh
# Clone the repository (with submodules)
git clone --recurse-submodules <repo-url>
cd eigcov

# If you already cloned without submodules, initialize them with:
#   git submodule update --init --recursive

# Vision & language experiments
UV_PROJECT_ENVIRONMENT=.venv-vl uv sync --group vision-language

# OLMo experiments
UV_PROJECT_ENVIRONMENT=.venv-olmo uv sync --group olmo

# Set env vars
export PYTHONPATH="$PYTHONPATH:$(pwd)" # Add src to python path
export HF_HOME=$SCRATCH/huggingface
# python -c "import nltk; nltk.download('punkt', download_dir='$SCRATCH/nltk_data')"
export NLTK_DATA=$SCRATCH/nltk_data
# TamIA: 
# export HF_HUB_DISABLE_HF_TRANSFER=1
# module load cuda/12.6 arrow python/3.11 httpproxy
```

## Vision Experiments (ViT-B-16 / ViT-B-32 / ViT-L-14)

```sh
# 1. Download checkpoints
gdown <URL> # or finetune using bash scripts/vision/finetune.sh
# 2. (Optional) Evaluate experts
bash scripts/vision/eval_single_task.sh
# 3. Evaluate merged models
bash scripts/vision/eval_task_addition.sh
```

Results are saved to `results/{model}-{method}/metrics.json`.

## Language Experiments (T5-Base / T5-Large)

```sh
# 1. Download checkpoints
gdown <URL>  # or finetune using bash scripts/vision/finetune.sh
# 2. (Optional) Evaluate experts
bash scripts/language/eval_single_task.sh
# 3. Evaluate merged models
bash scripts/language/eval_task_addition.sh
```

Results are saved to `results/{model}-{method}/metrics.json`.

## OLMo Experiments (Olmo-3-7b)

```sh
# 1. Download checkpoints
bash scripts/olmo/download_models.sh
# 2. (Optional) Evaluate experts
bash scripts/olmo/eval_single_task.sh
# 3. Evaluate merged models
bash scripts/olmo/eval_task_addition.sh # (default gpus: 4)
```


## Reproducing Plots
```sh

# Fig. 1-3 (cross|corr|drift) errors
# run finetune using --grad
```

