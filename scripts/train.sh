# Finetune using "standard", "lora", "linear" finetuning modes
for ft in standard lora linear; do
    python scripts/vision/finetune.py \
    --finetuning-mode="$ft" \
    --model=ViT-B-16 \
    --world-size=1 \
    --num-workers=1 \
    --openclip-cachedir=$SCRATCH/openclip \
    --data-location=$SLURM_TMPDIR/datasets