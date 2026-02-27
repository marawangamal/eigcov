#!/bin/bash
#SBATCH --job-name=ViT-L-14-standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4        # Keep as 1; your script spawns the others
#SBATCH --gres=gpu:rtx8000:4       # Specifically request 2 RTX 8000 GPUs
#SBATCH --cpus-per-task=4          # 4 CPUs per GPU is ideal
#SBATCH --mem=32G                  # Total system memory


# 1. Setup Data
source $SCRATCH/tangent_task_arithmetic/.venv/bin/activate
cd $SCRATCH/tangent_task_arithmetic
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

echo "Preparing data in $SLURM_TMPDIR..."
unzip -q $SCRATCH/tangent_task_arithmetic/vit_datasets_08.zip -d $SLURM_TMPDIR

# 2. Training
# Options: ("Cars" "DTD" "EuroSAT" "GTSRB" "MNIST" "RESISC45" "SUN397" "SVHN")
python src/finetune.py \
    --finetuning-mode=standard \
    --model=ViT-L-14 \
    --world-size=4 \
    --num-workers=1 \
    --openclip-cachedir=$SCRATCH/openclip \
    --data-location=$SLURM_TMPDIR/datasets 