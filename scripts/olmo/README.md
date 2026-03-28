# Olmo Experiments 

Merge 3 RL-Zero finetuned experts

## Setup

```sh
# # olmes (evaluation framework)
# git clone https://github.com/allenai/olmes.git
# cd olmes && uv sync && uv sync --group gpu
module load cuda/12.6 arrow python/3.12 httpproxy
cd ../olmes-fixed
source .venv/bin/activate
cd ../eigcov
export HF_HOME=$SCRATCH/huggingface
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs
# python -c "import nltk; nltk.download('punkt', download_dir='$SCRATCH/nltk_data')"
export NLTK_DATA=$SCRATCH/nltk_data
```

## Evaluate merge methods

```sh
bash scripts/olmo/eval_task_addition.sh \
  --pretrained-dir allenai/Olmo-3-1025-7B \
  --finetuned-dirs \
    allenai/Olmo-3-7B-RL-Zero-Math \
    allenai/Olmo-3-7B-RL-Zero-Code \
    allenai/Olmo-3-7B-RL-Zero-IF \
  --merge-funcs "eigcov tsv mean isoc" \
  --gpus 4
```

## Evaluate experts
```sh
MODEL_ID=allenai/Olmo-3-7B-RL-Zero-Math
  olmes \
    --model $MODEL_ID \
    --task \
      codex_humaneval::tulu \
      codex_humanevalplus::tulu \
      ifeval::tulu \
      aime:2024::olmo3:midtrain \
      aime:2025::olmo3:midtrain \
    --output-dir results-rl/"${MODEL_ID}/$(echo "$dir" | tr '/' '-')" \
    --gpus 4 \
    --model-type vllm \
    --model-args '{"gpu_memory_utilization": 0.8, "trust_remote_code": false, "max_length": 4096}' \
    --batch-size 128
```

## Collect results
```sh
# 5. Collect results
python scripts/olmo/collect_results.py --dirs results-rl/meta-llama-Meta-Llama-3.1-8B-eigcov
```


