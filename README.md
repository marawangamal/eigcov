# Covariance Estimation using Task Matrices

## Setup

```sh
pip install -r requirements.txt
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
cp vit_datasets_08.zip $SLURM_TMPDIR/
unzip -q $SLURM_TMPDIR/vit_datasets_08.zip -d $SLURM_TMPDIR/
```

## Vision Experiments (ViT-B-16 / ViT-B-32 / ViT-L-14)

### 1. Fine-tune

```sh
python scripts/vision/finetune.py \
  --finetuning-mode=standard \
  --model=ViT-B-16 \
  --world-size=1 \
  --num-workers=1 \
  --openclip-cachedir=$SCRATCH/openclip \
  --data-location=$SLURM_TMPDIR/datasets \
  --save=$SCRATCH/eigcov/checkpoints/vision 
```

### 2. Evaluate single model (zeroshot / standard / linear / lora)
```sh
python scripts/vision/eval_single_task.py \
  --finetuning-mode=standard \
  --model=ViT-B-32 \
  --openclip-cachedir=$SCRATCH/openclip \
  --data-location=$SLURM_TMPDIR/datasets
```

### 3. Evaluate merged models
**NOTE:** Must run `eval_single_task.py` in (2) with zeroshot and standard before merging methods can be evaluated using the commands below.

```sh
# EigenCov (data-free)
python scripts/vision/eval_task_addition.py \
  --model=ViT-B-16 --finetuning-mode=standard --merge-func=eigcov

# RegMean (must run covariance.py first, see below)
python scripts/vision/eval_task_addition.py \
  --model=ViT-B-16 --finetuning-mode=lora --merge-func=regmean \
  --mha=split \
  --cov-dir=results/ViT-B-16/covariances_strain_n10_b32_tsm_attnsplit_efull_ftlora \
  --coeff-start=0 --n-eval-points=11
```

## Scripts 

### Generate covariance matrices (only required for RegMean)
```sh
python scripts/vision/covariance.py \
  --model=ViT-B-32 \
  --cov-split=train \
  --cov-num-batches=10 \
  --cov-batch-size=32 \
  --mha=split \
  --cov-type=sm \
  --cov-estimator=full
```
**NOTE:** `--mha`: `split` Splits q,k,v into separate linear modules so their activation covariances can be collected (otherwise will be ignored).

### Generate covariance matrices at multiple sample counts
```sh
python scripts/vision/covariance.py \
  --model=ViT-B-16 \
  --cov-split=train \
  --cov-num-batches=1,10,100,500,1000 \
  --cov-batch-size=1 \
  --mha=split \
  --cov-type=sm \
  --cov-estimator=full \
  --openclip-cachedir=$SCRATCH/openclip \
  --data-location=$SLURM_TMPDIR/datasets
```


### Reproducing Vision Experiments
To reproduce vision experiments simply run the following commands. The results will be saved to `results/results.jsonl`
```sh
bash scripts/vision/vision_train.sh
bash scripts/vision/vision_eval.sh
```

### Generate gradient cross-term matrices
```sh
python scripts/vision/finetune.py \
    --model ViT-B-16 \
    --finetuning-mode standard \
    --batch-size 1 \
    --epochs 2 \
    --grad-cross-matrix \
    --data-location $SLURM_TMPDIR/datasets \
    --save checkpoints-accum-e2-v2-2026-06-15 \
    --openclip-cachedir $SCRATCH/openclip \
    --max-steps 10
```

## Language Experiments
### 1. Fine-tune
```sh
python scripts/language/finetune.py \
  --finetuning-mode=standard \
  --model=t5-base \
  --save=$SCRATCH/eigcov/checkpoints \
  --hf-cache-dir=$SCRATCH/hf_cache
```

### 2. Evaluate single model (zeroshot / standard / linear / lora)
```sh
python scripts/language/eval_single_task.py \
--finetuning-mode=standard  --save=$SCRATCH/eigcov/checkpoints
```

### 3. Evaluate merged models
**NOTE:** Must run `eval_single_task.py` in (2) with zeroshot and standard before merging methods can be evaluated using the commands below.

```sh
python scripts/language/eval_task_addition.py \
  --model=t5-base --finetuning-mode=standard --merge-func=eigcov --save=$SCRATCH/eigcov/checkpoints/language
```

### Generate covariance matrices at multiple sample counts
```sh
python scripts/language/covariance.py \
  --model=t5-large \
  --cov-split=train \
  --cov-num-batches=1,10,100,500,1000 \
  --cov-batch-size=1 \
  --cov-type=sm \
  --cov-estimator=full 
```



## NLG Experiments

### 1. Fine-tune
```sh
# Setup.
module load cuda/12.6 arrow python/3.12 httpproxy
export HF_HOME=$SCRATCH/huggingface
source .venv/bin/activate

# Multi-GPU full fine-tune with FSDP (required for 8B full fine-tune)
torchrun --nproc_per_node=4 scripts/nlg/finetune.py \
  --capability math --fsdp \
  --output-dir $SCRATCH/eigcov/checkpoints/nlg \
  --save-strategy steps --save-steps 200

torchrun --nproc_per_node=4 scripts/nlg/finetune.py \
  --capability general --fsdp \
  --output-dir $SCRATCH/eigcov/checkpoints/nlg \
  --save-strategy steps

```

### Setup (olmes evaluation)
```bash
module load cuda/12.6 arrow python/3.12 httpproxy
git clone https://github.com/allenai/olmes.git
cd olmes
uv sync
uv sync --group gpu # for vLLM support
```


## Repository Structure

```
src/                          # Library code (importable modules)
├── task_vectors.py           # Task vector arithmetic (model-agnostic)
├── merging.py                # Merging strategies: EigenCov, RegMean, TSV, ...
├── covariance.py             # OnlineCovariance + register_hooks (model-agnostic)
├── distributed.py            # DDP utilities (model-agnostic)
├── mhap.py / mhas.py         # Custom multi-head attention variants: packed / split (model-agnostic)
├── args.py                   # Shared argument parser
├── utils.py                  # Shared utilities
│
├── vision/                   # Vision-specific library code (OpenCLIP / ViT)
│   ├── modeling.py           # ImageEncoder, ImageClassifier
│   ├── heads.py              # Zero-shot classification heads
│   ├── eval.py               # eval_single_dataset, evaluate_task_vector, ...
│   ├── linearize.py          # Taylor-linearized vision encoder
│   └── datasets/             # MNIST, Cars, DTD, EuroSAT, GTSRB, ...
│
└── language/                 # Language-specific code
    ├── modeling.py
    ├── linearize.py
    ├── args.py
    ├── eval/
    └── datasets/

scripts/                      # Entry points (run directly)
├── setup.sh
├── vision/
│   ├── finetune.py           # Fine-tune vision models
│   ├── eval_single_task.py   # Evaluate single fine-tuned model
│   ├── eval_task_addition.py # Evaluate merged model
│   ├── eval_task_negation.py
│   ├── covariance.py         # Collect per-layer covariance matrices
│   └── correlation.py        # Collect correlations of actiavations & gradients
└── language/
    ├── finetune.py           # Fine-tune T5 models
    ├── eval_single_task.py   # Evaluate single fine-tuned model
    ├── eval_task_addition.py
    └── eval_task_negation.py
```