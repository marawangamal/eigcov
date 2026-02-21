# Covariance Estimation using Task Matrices

## Quickstart
To run the code, please install all its dependencies:
```sh
# 1. Setup environment
pip install -r requirements.txt
export PYTHONPATH="$PYTHONPATH:$PWD"
export SSL_CERT_DIR=/etc/ssl/certs

# 2. Finetune
cp vit_datasets_08.zip $SLURM_TMPDIR
cd $SLURM_TMPDIR && unzip vit_datasets_08.zip -d ./ && cd -
# options: standard,lora
python src/finetune.py --finetuning-mode=lora --model=ViT-B-32 --world-size=1 --openclip-cachedir=$SCRATCH/openclip --data-location=$SLURM_TMPDIR/datasets --train-dataset=SUN397

# 3. Evaluate single task (none,standard,lora)
python src/eval_single_task.py --model=ViT-B-32 --finetuning-mode=standard --openclip-cachedir=$SCRATCH/openclip --data-location=$SLURM_TMPDIR/datasets

# 4. Evaluate merged (none,standard,lora)
python src/eval_task_addition.py --model=ViT-B-32 --finetuning-mode=lora --openclip-cachedir=$SCRATCH/openclip --data-location=datasets --merge-func=regmean --coeff-start=1.0 --n-eval-points=1

# 5. Plots:
python scripts/decorrelation_offline.py --model=ViT-B-16 --openclip-cachedir=$SCRATCH/openclip --data-location=$SLURM_TMPDIR/datasets
python scripts/disentanglement.py --model=ViT-B-16 --openclip-cachedir=$SCRATCH/openclip --data-location=$SLURM_TMPDIR/datasets/datasets
python scripts/interference.py --model=ViT-B-16 --openclip-cachedir=$SCRATCH/openclip --data-location=$SLURM_TMPDIR/datasets
```

## Repository content

This repository is heavily based on the code from [Ilharco et al. (2022)](https://github.com/mlfoundations/task_vectors) and follows the same structure.

### Training

The script `src/finetune.py` can be used to reproduce the training protocol we used to fine-tune our models on all our downstream tasks (both linearly and non-linearly).
```sh 
python src/finetune.py --finetuning-mode=standard --model=ViT-B-32 --world-size=2 # Finetune non-linearly on 2 GPUs
python src/finetune.py --finetuning-mode=linear --model=ViT-B-32 --world-size=2 # Finetune non-linearly on 2 GPUs
```

### Evaluation

We provide different scripts to evaluate the different task vectors obtained using the previous scripts.

#### Single-task accuracy
Having run `src/finetune.py` for a given model, you can evaluate the performance of the fine-tuned weights on each single task by running
```sh 
# Evaluate pre-trained models.
python src/eval_single_task.py --model=ViT-B-32 --finetuning-mode=none

# Evaluate non-linearly fine-tuned models.
python src/eval_single_task.py --model=ViT-B-32 --finetuning-mode=standard

# Evaluate linearly fine-tuned models.
python src/eval_single_task.py --model=ViT-B-32 --finetuning-mode=linear

# Evaluate post-hoc linearized models. Requires having run finetune.py with --finetuning=mode=standard.
python src/eval_single_task.py --model=ViT-B-32 --finetuning-mode=posthoc
```

#### Task addition
Once evaluated on the single tasks, we can evaluate the task arithmetic performance of the different strategies on the addition benchmark.
```sh 
# Evaluate non-linearly fine-tuned models.
python src/eval_task_addition.py --model=ViT-B-32 --finetuning-mode=standard

# Evaluate linearly fine-tuned models.
python src/eval_task_addition.py --model=ViT-B-32 --finetuning-mode=linear

# Evaluate post-hoc linearized models.
python src/eval_task_addition.py --model=ViT-B-32 --finetuning-mode=posthoc
```

## Datasets
To download and prepare the datasets, please follow the instructions in [this issue](https://github.com/mlfoundations/task_vectors/issues/1).

## Reference
If you find this code useful, please cite the following paper:
```bibtex
@article{ortizjimenez2023tangent,
  title   = {Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained
             Models},
  author  = {Guillermo Ortiz{-}Jim{\'{e}}nez and
             Alessandro Favero and
             Pascal Frossard},
  journal = {arXiv:2305.12827},
  year    = {2023},
  note    = {\url{https://arxiv.org/abs/2305:12827}},
}
```

