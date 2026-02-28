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
└── language/                # Language-specific library code (T5 / HuggingFace)
    ├── modeling.py
    ├── linearize.py
    ├── args.py
    ├── eval/
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
└── language/
    ├── finetune.py          # Fine-tune T5 models
    ├── eval_single_task.py  # Evaluate single fine-tuned model
    ├── eval_task_addition.py
    └── eval_task_negation.py
```

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

<!-- salloc --gres=gpu:rtx8000:4 --ntasks-per-node=1 --cpus-per-task=8 --mem=32G -->

```sh
python scripts/vision/finetune.py \
  --finetuning-mode=lora \
  --model=ViT-B-32 \
  --world-size=1 \
  --num-workers=1 \
  --openclip-cachedir=$SCRATCH/openclip \
  --data-location=$SLURM_TMPDIR/datasets
```

Options for `--finetuning-mode`: `standard`, `lora`, `linear`, `posthoc`.

### 2. Evaluate single task (zeroshot / standard / linear / lora)

```sh
python scripts/vision/eval_single_task.py \
  --finetuning-mode=lora \
  --model=ViT-B-32 \
  --openclip-cachedir=$SCRATCH/openclip \
  --data-location=$SLURM_TMPDIR/datasets
```

### 3. Collect covariance matrices

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
`--mha`: `split` Splits q,k,v into separate linear modules.

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
  --model=ViT-B-16 --finetuning-mode=lora --merge-func=regmean \
  --mha=split \
  --cov-dir=results/ViT-B-16/covariances_strain_n10_b32_tsm_attnsplit_efull_ftlora \
  --coeff-start=0 --n-eval-points=11



python scripts/vision/eval_task_addition.py  --model=ViT-L-14 --finetuning-mode=standard --merge-func=regmean \
--coeff-start=1.0 --n-eval-points=1 --mha=split --cov-dir results/ViT-L-14/covariances_strain_n10_b32_tsm_attnsplit_efull
# next


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

## Language Experiments (t5-base)

Datasets: `qasc`, `wiki_qa`, `quartz`, `paws`, `story_cloze`, `winogrande`, `wsc`

### 1. Fine-tune

```sh
python scripts/language/finetune.py \
  --finetuning-mode=standard \
  --model=t5-base \
  --save=$SCRATCH/eigcov/checkpoints/language \
  --hf-cache-dir=$SCRATCH/hf_cache
```

Options for `--finetuning-mode`: `standard`, `linear`.

### 2. Evaluate zero-shot / standard / Linear

```sh
# Zero-shot (pretrained model, no task vector applied)
python scripts/language/eval_single_task.py \
  --finetuning-mode=none \
  --save=$SCRATCH/eigcov/checkpoints/language

# Fine-tuned
python scripts/language/eval_single_task.py \
  --finetuning-mode=standard \
  --save=$SCRATCH/eigcov/checkpoints/language

# Linear
python scripts/language/eval_single_task.py \
  --finetuning-mode=linear \
  --save=$SCRATCH/eigcov/checkpoints/language
```

Saves results to `{save}/zeroshot_accuracies.json` or `{save}/ft_accuracies.json`.

### 3. Evaluate task addition (task arithmetic)

Requires `zeroshot_accuracies.json` and `ft_accuracies.json` (or `linear_ft_accuracies.json`) to exist in `--save`.

```sh
python scripts/language/eval_task_addition.py \
  --finetuning-mode=standard \
  --save=$SCRATCH/eigcov/checkpoints/language
```

Saves results to `{save}/additions.json`.

### 4. Evaluate task negation

```sh
python scripts/language/eval_task_negation.py \
  --finetuning-mode=standard \
  --save=$SCRATCH/eigcov/checkpoints/language \
  --control-threshold=0.95
```

Uses `rte` as the control dataset. Saves results to `{save}/negations.json`.

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


## Overview

**Model: T5**(encoder-decoder)                                                                                                                                                  

t5-base is a seq2seq (encoder-decoder) transformer. The encoder reads the input prompt, the decoder generates the output. It's NOT a causal/decoder-only LM like GPT — it was
  pretrained on span-corruption (masking spans and predicting them), but the lm-adapt variants were further trained on language modeling.                                     
                

**What's being tested**

All 7 datasets (qasc, wiki_qa, quartz, paws, story_cloze, winogrande, wsc) are multiple-choice classification tasks. During eval, the model doesn't generate text — instead
it scores each candidate answer by computing the log probability of the decoder producing that answer string given the encoded input:

score(choice_i) = sum of log P(token | context) over all tokens in choice_i

The predicted answer is argmax over all choices. This is called rank classification or closed-form eval — much faster and more reliable than generation.

During training, it's standard cross-entropy on the correct answer text (teacher forcing).


**Prompt templating: promptsource, not HF chat templates**

It's completely different from HF chat templates. promptsource (from the BigScience T0 paper) is a library of hand-written Jinja2 templates that convert structured dataset
fields into natural language. Each dataset has multiple templates written by crowd workers.

For example, for qasc a template might look like:

**Jinja2 template**

{{fact1}} {{fact2}} Based on these facts, {{question}}
- {{choices[0]}}
- {{choices[1]}}
...
Answer: ||| {{choices[answer_idx]}}

The ||| separator splits input from target. So for a concrete example:

Input:  "Magnets attract certain metals. Iron is a metal. Based on these facts,
          what do magnets attract? - Iron - Wood - Plastic"
Target: "Iron"

HF chat templates in contrast add structural tokens (<|system|>, <|user|>, <|assistant|>) and are designed for instruction-tuned conversational models. Promptsource
templates are purely natural language reformulations with no special tokens — the task is framed as a fill-in-the-blank text completion problem, which fits T5's seq2seq
pretraining naturally.