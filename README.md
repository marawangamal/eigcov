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
  --save-strategy steps --save-steps 200 --resume

torchrun --nproc_per_node=4 scripts/nlg/finetune.py \
  --capability math --fsdp \
  --output-dir $SCRATCH/eigcov/checkpoints/nlg-lora \
  --hf-cache-dir $SCRATCH/huggingface \
  --use-lora \
  --save-strategy steps --save-steps 200
  ```

### 2. Merge
```sh
# Create param folders
python scripts/nlg/save_model_param_folder.py --model mremila/Llama-3.1-8B-coding     --output-dir checkpoints/nlg/mremila-Llama-3.1-8B-coding
python scripts/nlg/save_model_param_folder.py --model mremila/Llama-3.1-8B-precise_if --output-dir checkpoints/nlg/mremila-Llama-3.1-8B-precise_if
python scripts/nlg/save_model_param_folder.py --model mremila/Llama-3.1-8B-general    --output-dir checkpoints/nlg/mremila-Llama-3.1-8B-general
python scripts/nlg/save_model_param_folder.py --model mremila/Llama-3.1-8B-knowledge  --output-dir checkpoints/nlg/mremila-Llama-3.1-8B-knowledge
# ... do for all

# Merge param folder (lora)
method=tsv
python scripts/nlg/merge.py \
  --pretrained-dir checkpoints/nlg/meta-llama-Meta-Llama-3.1-8B \
  --finetuned-dirs \
    checkpoints/nlg/mremila-Llama-3.1-8B-math \
    checkpoints/nlg/mremila-Llama-3.1-8B-coding \
    checkpoints/nlg/mremila-Llama-3.1-8B-precise-if \
    checkpoints/nlg/mremila-Llama-3.1-8B-general \
    checkpoints/nlg/mremila-Llama-3.1-8B-knowledge \
  --merge-func $method \
  --output-dir checkpoints/nlg/mremila-Llama-3.1-8B-${method} 

# Merge param folder (ignore)
# ignore_keys="gate_proj up_proj"
method=eigcov_gd
python scripts/nlg/merge.py \
  --pretrained-dir checkpoints/nlg/meta-llama-Meta-Llama-3.1-8B \
  --finetuned-dirs \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-math-reasoning \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-coding \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-precise-if \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-general \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-knowledge-recall \
  --merge-func $method \
  --output-dir checkpoints/nlg/pmahdavi-Llama-3.1-8B-${method}
  # --output-dir checkpoints/nlg/pmahdavi-Llama-3.1-8B-${method}-ignore-${ignore_keys// /-} \
  # --ignore-keys $ignore_keys

# Merge param folder (mix: two methods per-key)
python scripts/nlg/merge.py \
  --pretrained-dir checkpoints/nlg/meta-llama-Meta-Llama-3.1-8B \
  --finetuned-dirs \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-math-reasoning \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-coding \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-precise-if \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-general \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-knowledge-recall \
  --merge-func mix \
  --merge-kwargs '{"mix_primary": "eigcov_gd", "mix_fallback": "eigcov", "mix_targets": ["down_proj", "gate_proj", "up_proj"]}' \
  --output-dir checkpoints/nlg/pmahdavi-Llama-3.1-8B-mix-eigcov_gd-eigcov_down_proj_gate_proj_up_proj
```

### 3. Upload 
```bash
# 
hf upload mremila/Llama-3.1-8B-math checkpoints/nlg/Llama-3.1-8B-math --repo-type model                                            
hf upload mremila/Llama-3.1-8B-coding checkpoints/nlg/Llama-3.1-8B-coding --repo-type model                                        
hf upload mremila/Llama-3.1-8B-general checkpoints/nlg/Llama-3.1-8B-general --repo-type model                                      
hf upload mremila/Llama-3.1-8B-knowledge checkpoints/nlg/Llama-3.1-8B-knowledge --repo-type model                                  
hf upload mremila/Llama-3.1-8B-precise_if checkpoints/nlg/Llama-3.1-8B-precise_if --repo-type model    

# Merged models
method=isoc
echo "Uploading merged model for method: $method"
hf upload mremila/pmahdavi-Llama-3.1-8B-$method checkpoints/nlg/pmahdavi-Llama-3.1-8B-$method --repo-type model
echo "Upload complete to remote repo: mremila/pmahdavi-Llama-3.1-8B-$method"


hf upload mremila/pmahdavi-Llama-3.1-8B-eigcov-ignore-gate_proj-up_proj checkpoints/nlg/pmahdavi-Llama-3.1-8B-eigcov-ignore-gate_proj-up_proj --repo-type model

# 
```

### 4. Evaluate 
```bash

## Setup (olmes evaluation)
# module load cuda/12.6 arrow python/3.12 httpproxy
# git clone https://github.com/allenai/olmes.git
# cd olmes
# uv sync
# uv sync --group gpu # for vLLM support

## Run Evaluation
method=isoc 
olmes --model mremila/pmahdavi-Llama-3.1-8B-$method \
--task  codex_humaneval::tulu codex_humanevalplus::tulu \
gsm8k::tulu drop::llama3 minerva_math::tulu  \
ifeval::tulu popqa::tulu "bbh:cot-v1::tulu" \
--output-dir results-nlg-4096-$method \
--gpus 4 \
--model-type vllm \
--model-args '{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 4096}' 


## Run Evaluation (20K)
method=isoc 
olmes --model mremila/pmahdavi-Llama-3.1-8B-20K-$method \
--task  codex_humaneval::tulu codex_humanevalplus::tulu \
gsm8k::tulu drop::llama3 minerva_math::tulu  \
ifeval::tulu popqa::tulu "bbh:cot-v1::tulu" \
--output-dir results-nlg-4096-20K-$method \
--gpus 2 \
--model-type vllm \
--model-args '{"gpu_memory_utilization": 0.9, "trust_remote_code": false, "max_length": 4096}' 



#  97/3280 [06:27<3:03:59,  3.47s/it]
olmes \
  --model mremila/pmahdavi-Llama-3.1-8B-mix-eigcov_gd-eigcov_down_proj_gate_proj_up_proj \
  --task  codex_humaneval::tulu codex_humanevalplus::tulu \
  gsm8k::tulu drop::llama3 minerva_math::tulu  \
  ifeval::tulu popqa::tulu "bbh:cot-v1::tulu" \
  --output-dir results-nlg-mix-eigcov_gd-eigcov_down_proj_gate_proj_up_proj \
  --gpus 1 \
  --batch-size 128 \
  --model-type vllm  \
  --model-args '{"gpu_memory_utilization": 0.9, "trust_remote_code": false, "max_length": 4096}' 


olmes \
--model mremila/pmahdavi-Llama-3.1-8B-eigcov-ignore-gate_proj-up_proj \
--model-type vllm \
--model-args '{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 4096}' \
--task  codex_humaneval::tulu codex_humanevalplus::tulu \
gsm8k::tulu drop::llama3 minerva_math::tulu  \
ifeval::tulu popqa::tulu "bbh:cot-v1::tulu" \
--num-workers 1 \
--batch-size 128 \
--output-dir results-nlg-eigcov-ignore-gate_proj-up_proj


# code only
olmes \
  --model mremila/Llama-3.1-8B-coding \
  --task codex_humaneval::tulu    \
  --output-dir results-nlg-mremila-Llama-3.1-8B-coding \
  --gpus 4 \
  --model-type vllm  \
  --batch-size 256 \
  --model-args '{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 4096}' 
```
<!-- 113/3280 [30:57<23:38:53, 26.88s/it] w/ 1 gpu -->
<!-- codex_humanevalplus::tulu -->


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







<!-- Running generate_until requests:  66%|█████████████████████████████████████████████████████████████████████████████████████████████████▉                                                  | 9439/14267 [06:02<03:53, 20.71it/s]
Running generate_until requests:  66%|█████████████████████████████████████████████████████████████████████████████████████████████████▉                                                  | 9445/14267 [06:02<03:27, 23.22it/s]
Running generate_until requests:  68%|████████████████████████████████████████████████████████████████████████████████████████████████████▍                                               | 9682/14267 [06:12<03:08, 24.31it/s]^C(Worker_TP1 pid=1864815) INFO 03-25 12:02:48 [multiproc_executor.py:558] Parent process exited, terminating worker
(Worker_TP0 pid=1864814) INFO 03-25 12:02:48 [multiproc_executor.py:558] Parent process exited, terminating worker
(Worker_TP3 pid=1864817) INFO 03-25 12:02:48 [multiproc_executor.py:558] Parent process exited, terminating worker
(Worker_TP2 pid=1864816) INFO 03-25 12:02:48 [multiproc_executor.py:558] Parent process exited, terminating worker -->