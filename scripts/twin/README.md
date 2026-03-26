# Twin-Merging Experiments (Qwen-14B)

Merge 4 LoRA adapters (`lu-vae/qwen-{bbq,cnn,mmlu,truthfulqa}-merged`) with eigcov/baselines and evaluate on BBQ, CNN/DailyMail, MMLU, TruthfulQA.

## Setup

Qwen-1 requires `einops` and a dummy `transformers_stream_generator` stub (the real package is incompatible with newer `transformers`):

```sh
pip install einops peft safetensors lm-eval
mkdir -p $VIRTUAL_ENV/lib/python*/site-packages/transformers_stream_generator
echo "" > $VIRTUAL_ENV/lib/python*/site-packages/transformers_stream_generator/__init__.py
```

## Full pipeline

```sh
bash scripts/twin/run_all.sh
```

## Step-by-step

```sh
# 1. Convert base model to param-folder
python scripts/nlg/save_model_param_folder.py --model Qwen/Qwen-14B --output-dir checkpoints/twin/Qwen-14B

# 2. Convert LoRA adapters to param-folder
python scripts/twin/save_lora_param_folder.py --base-model Qwen/Qwen-14B --adapter lu-vae/qwen-bbq-merged --output-dir checkpoints/twin/qwen-bbq-merged
python scripts/twin/save_lora_param_folder.py --base-model Qwen/Qwen-14B --adapter lu-vae/qwen-cnn-merged --output-dir checkpoints/twin/qwen-cnn-merged
python scripts/twin/save_lora_param_folder.py --base-model Qwen/Qwen-14B --adapter lu-vae/qwen-mmlu-merged --output-dir checkpoints/twin/qwen-mmlu-merged
python scripts/twin/save_lora_param_folder.py --base-model Qwen/Qwen-14B --adapter lu-vae/qwen-truthfulqa-merged --output-dir checkpoints/twin/qwen-truthfulqa-merged

# 3. Merge (replace eigcov with mean/tsv/isoc for baselines)
method=mean
python scripts/nlg/merge.py \
    --pretrained-dir checkpoints/twin/Qwen-14B \
    --finetuned-dirs \
        checkpoints/twin/qwen-bbq-merged \
        checkpoints/twin/qwen-cnn-merged \
        checkpoints/twin/qwen-mmlu-merged \
        checkpoints/twin/qwen-truthfulqa-merged \
    --merge-func $method \
    --output-dir checkpoints/twin/Qwen-14B-$method \
    --trust-remote-code

# 4. Evaluate
bash scripts/twin/eval.sh checkpoints/twin/Qwen-14B-mean results/twin/mean

# 5. Collect results
python scripts/twin/collect_results.py --dirs results/twin/eigcov results/twin/mean results/twin/tsv results/twin/isoc
```

