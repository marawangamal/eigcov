import torch

from src.language.args import parse_arguments
from src.language.finetune import finetune

T5_DATASETS = ["qasc", "wiki_qa", "quartz", "paws", "story_cloze", "winogrande", "wsc"]

args = parse_arguments()

# Set default training hyperparameters matching the talos paper
args.lr = 1e-4
args.wd = 0.0
args.batch_size = 64
args.num_grad_accumulation = 16
args.max_seq_len = 128
args.num_batches = 75000

for dataset in T5_DATASETS:
    args.train_dataset = dataset

    print("=" * 100)
    print(f"Fine-tuning {args.model} on {dataset} ({args.finetuning_mode})")
    print("=" * 100)

    torch.multiprocessing.spawn(finetune, args=(args,), nprocs=args.world_size)
