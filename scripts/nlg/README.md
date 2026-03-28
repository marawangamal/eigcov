# NLG Experiments (Llama-3.1-8B)

Merge 5 capability-specialized LoRA fine-tunes and evaluate on BBH-CoT, HumanEval(+), DROP, GSM8K, IFEval, MATH, PopQA.

## Setup

```sh
# olmes (evaluation framework)
git clone https://github.com/allenai/olmes.git
cd olmes && uv sync && uv sync --group gpu
source .venv/bin/activate
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
python -c "import nltk; nltk.download('punkt', download_dir='$SCRATCH/nltk_data')"
export NLTK_DATA=$SCRATCH/nltk_data
```

## Full pipeline

```sh
bash scripts/nlg/eval_task_addition.sh \
  --pretrained-dir checkpoints/nlg/meta-llama-Meta-Llama-3.1-8B \
  --finetuned-dirs \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-math-reasoning \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-coding \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-precise-if \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-general \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-knowledge-recall \
  --merge-funcs "eigcov_gd" \
  --merge-kwargs '{"lr": 1e-5, "max_iters": 300, "alpha_weighted": true}' \
  --gpus 4
```

## Step-by-step

```sh
# 1. Fine-tune on each capability (single GPU w/ LoRA, or multi-GPU w/ FSDP)
python scripts/nlg/finetune.py --capability math --use-lora --hf-cache-dir $SCRATCH/huggingface

# 2. Convert pretrained + finetuned models to param-folder format
python scripts/nlg/save_model_param_folder.py --model meta-llama/Meta-Llama-3.1-8B \
  --output-dir checkpoints/nlg/meta-llama-Meta-Llama-3.1-8B

# 3. Merge
python scripts/nlg/merge.py \
  --pretrained-dir checkpoints/nlg/meta-llama-Meta-Llama-3.1-8B \
  --finetuned-dirs \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-math-reasoning \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-coding \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-precise-if \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-general \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-knowledge-recall \
  --merge-func eigcov \
  --output-dir checkpoints/nlg/meta-llama-Meta-Llama-3.1-8B-eigcov

# 4. Evaluate
olmes --model checkpoints/nlg/meta-llama-Meta-Llama-3.1-8B-eigcov \
  --task codex_humaneval::tulu codex_humanevalplus::tulu \
    gsm8k::tulu drop::llama3 minerva_math::tulu \
    ifeval::tulu popqa::tulu "bbh:cot-v1::tulu" \
  --output-dir results-nlg/meta-llama-Meta-Llama-3.1-8B-eigcov \
  --gpus 4 --model-type vllm \
  --model-args '{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 4096}'

# 5. Collect results
python scripts/nlg/collect_results.py --dirs results-nlg/meta-llama-Meta-Llama-3.1-8B-eigcov
```

## Custom merge kwargs

Pass `--merge-kwargs` with a JSON dict to forward extra arguments to the merge function. For example, the `mix` merge uses a primary function on target layers and a fallback on the rest:

```sh
bash scripts/nlg/eval_task_addition.sh \
  --pretrained-dir checkpoints/nlg/meta-llama-Meta-Llama-3.1-8B \
  --finetuned-dirs \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-math-reasoning \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-coding \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-precise-if \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-general \
    checkpoints/nlg/pmahdavi-Llama-3.1-8B-knowledge-recall \
  --merge-funcs "mix" \
  --merge-kwargs '{"mix_primary": "eigcov_gd", "mix_fallback": "eigcov", "mix_targets": ["down_proj", "gate_proj", "up_proj"], "lr": }' \
  --gpus 4
# Output dir auto-named: meta-llama-Meta-Llama-3.1-8B-mix-eigcov_gd_eigcov_down_proj_gate_proj_up_proj
```

## Using HuggingFace model IDs directly

You can pass HF model IDs instead of local param-folder paths. The script auto-downloads and converts them to param-folder format (cached in `--output-base`):

```sh
bash scripts/nlg/eval_task_addition.sh \
  --pretrained-dir meta-llama/Meta-Llama-3.1-8B \
  --finetuned-dirs \
    pmahdavi/Llama-3.1-8B-math-reasoning \
    pmahdavi/Llama-3.1-8B-coding \
    pmahdavi/Llama-3.1-8B-precise-if \
    pmahdavi/Llama-3.1-8B-general \
    pmahdavi/Llama-3.1-8B-knowledge-recall \
  --merge-funcs "eigcov" \
  --gpus 4
```


## Olmo 3

You can pass HF model IDs instead of local param-folder paths. The script auto-downloads and converts them to param-folder format (cached in `--output-base`):

```sh
bash scripts/nlg/eval_task_addition.sh \
  --pretrained-dir allenai/Olmo-3-1025-7B \
  --finetuned-dirs \
    allenai/Olmo-3-7B-RL-Zero-Math \
    allenai/Olmo-3-7B-RL-Zero-Code \
    allenai/Olmo-3-7B-RL-Zero-IF \
  --merge-funcs "eigcov" \
  --gpus 4
```
