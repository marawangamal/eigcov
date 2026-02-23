# Covariance Estimation using Task Matrices

## Repository Structure

```
src/                         # Library code (importable modules)
├── task_vectors.py          # Task vector arithmetic (model-agnostic)
├── merging.py               # Merging strategies: sum, RegMean, EigenCov, PRM, ...
├── merging_ties.py          # TIES merging
├── covariance.py            # OnlineCovariance + register_hooks (model-agnostic)
├── distributed.py           # DDP utilities (model-agnostic)
├── mhap.py / mhas.py        # Custom MHA variants: packed / split (model-agnostic)
├── args.py                  # Shared argument parser
├── utils.py                 # Shared utilities
│
├── vision/                  # Vision-specific library code (OpenCLIP / ViT)
│   ├── modeling.py          # ImageEncoder, ImageClassifier
│   ├── heads.py             # Zero-shot classification heads
│   ├── eval.py              # eval_single_dataset, evaluate_task_vector, ...
│   ├── linearize.py         # Taylor-linearized vision encoder
│   └── datasets/            # MNIST, Cars, DTD, EuroSAT, GTSRB, ...
│
└── language/                # Language-specific library code (TODO)
    ├── modeling.py
    ├── eval.py
    └── datasets/

scripts/                     # Entry points (run directly)
├── setup.sh
├── vision/
│   ├── finetune.py          # Fine-tune vision models
│   ├── eval_single_task.py  # Evaluate single fine-tuned model
│   ├── eval_task_addition.py
│   ├── eval_task_negation.py
│   ├── covariance.py        # Collect per-layer covariance matrices
│   ├── decorrelation*.py    # Activation–gradient decorrelation analysis
│   ├── interference.py      # Per-layer interference analysis
│   ├── disentanglement.py
│   └── gmag.py
└── language/                # (TODO)
```

## Setup

```sh
pip install -r requirements.txt
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
```

## Vision Experiments (ViT-B-16 / ViT-B-32 / ViT-L-14)

### 1. Fine-tune

```sh
python scripts/vision/finetune.py \
  --finetuning-mode=standard \
  --model=ViT-B-16 \
  --world-size=4 \
  --num-workers=1 \
  --openclip-cachedir=$SCRATCH/openclip \
  --data-location=$SLURM_TMPDIR/datasets
```

Options for `--finetuning-mode`: `standard`, `lora`, `linear`, `posthoc`.

### 2. Evaluate single task

```sh
python scripts/vision/eval_single_task.py \
  --finetuning-mode=standard \
  --model=ViT-B-16 \
  --openclip-cachedir=$SCRATCH/openclip \
  --data-location=$SLURM_TMPDIR/datasets
```

### 3. Collect covariance matrices

```sh
# Sampled estimator (default, memory-efficient)
python scripts/covariance.py \
  --model=ViT-B-16 \
  --cov-split=train \
  --cov-num-batches=50 \
  --cov-batch-size=32 \
  --mha=split \
  --cov-estimator=sampled

# Full-sequence estimator
python scripts/covariance.py \
  --model=ViT-B-16 \
  --cov-split=train \
  --cov-num-batches=10 \
  --cov-batch-size=32 \
  --mha=split \
  --cov-type=sm \
  --cov-estimator=full
```

`--cov-type`: `cov` (centered covariance) or `sm` (uncentered second moment).
`--cov-estimator`: `sampled` (one random token per sample, Dx1) or `full` (entire sequence, DxT).

### 4. Evaluate merged models

```sh
# Task Arithmetic
python scripts/vision/eval_task_addition.py \
  --model=ViT-B-16 --finetuning-mode=standard --merge-func=sum

# EigenCov (data-free)
python scripts/vision/eval_task_addition.py \
  --model=ViT-B-16 --finetuning-mode=standard --merge-func=eigcov

# RegMean (requires covariance)
python scripts/vision/eval_task_addition.py \
  --model=ViT-B-16 --finetuning-mode=standard --merge-func=regmean \
  --mha=split \
  --cov-dir=results/ViT-B-16/covariances_strain_n50_b32_tcov_attnsplit_esampled

# Projected RegMean
python scripts/vision/eval_task_addition.py \
  --model=ViT-B-16 --finetuning-mode=standard --merge-func=prm \
  --coeff-start=1.0 --n-eval-points=1
```

### 5. Analysis scripts

```sh
python scripts/decorrelation_offline.py --model=ViT-B-16 ...
python scripts/interference.py --model=ViT-B-16 ...
python scripts/disentanglement.py --model=ViT-B-16 ...
```

## Language Experiments (TODO)

`src/language/` contains stubs for language model fine-tuning and evaluation.
Planned: HuggingFace-based fine-tuning on GLUE / SuperGLUE tasks with the
same task vector + merging pipeline.

## Covariance Estimation API

`OnlineCovariance` and `register_hooks` in `src/covariance.py` are
model-agnostic and work with any `nn.Linear` / `nn.MultiheadAttention`
layer. Pass modality-specific module types via `extra_module_types`:

```python
from src.covariance import register_hooks

cobjs, handles = register_hooks(
    model, args,
    extra_module_types=(MyCustomAttention,),
)
```
