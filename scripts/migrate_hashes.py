"""Recompute run_hash for all records in a results JSONL file.

After changing which args are included in the hash (e.g. adding an ignore
list), old records will have stale hashes.  This script rewrites every record
in-place with the new hash so that record_exists() deduplication works
correctly going forward.

Usage:
    python scripts/migrate_hashes.py                          # default: results/results.jsonl
    python scripts/migrate_hashes.py --db results/results.jsonl
    python scripts/migrate_hashes.py --dry-run               # print diffs, don't write
    python scripts/migrate_hashes.py --deduplicate           # drop later copy of each duplicate
    python scripts/migrate_hashes.py --fix-cov-dir          # patch buggy cov_dir paths before rehashing
"""

import argparse
import json
import shutil
from pathlib import Path

from src.results_db import make_run_hash

# Must match _HASH_IGNORE in scripts/vision/eval_task_addition.py and
# scripts/language/eval_task_addition.py.
_HASH_IGNORE = {
    # training-only
    "lr",
    "wd",
    "ls",
    "warmup_length",
    "epochs",
    "num_grad_accumulation",
    "batch_size",
    "checkpoint_every",
    "keep_checkpoints",
    "port",
    "world_size",
    "cosine_samples",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "lora_target_parameters",
    # environment / paths
    "openclip_cachedir",
    "hf_cache_dir",
    "cache_dir",
    "save",
    "data_location",
    # dynamically set after hash
    "eval_datasets",
    "finetuning_accuracies",
    "control_dataset",
    "eval_split",
    "eval_max_batches",
    # metadata
    "results_db",
    "exp_name",
    "overwrite",
    "num_workers",
    "device",
}

# Keys that are stored in the record but are NOT args (never included in hash).
_RECORD_ONLY_KEYS = {"run_hash", "timestamp"}


def record_to_args_dict(record):
    """Extract the args subset of a record (drop record-only and result keys)."""
    # Result keys are anything not in the original args — we identify args as
    # every key that is NOT a known result prefix and NOT a record-only key.
    skip_prefixes = ("test_", "val_", "optimal_coef")
    return {
        k: v
        for k, v in record.items()
        if k not in _RECORD_ONLY_KEYS
        and not any(k.startswith(p) for p in skip_prefixes)
    }


class _FakeNamespace:
    """Minimal stand-in for argparse.Namespace so make_run_hash can consume it."""

    def __init__(self, d):
        self.__dict__.update(d)


def canonical_cov_dir(model, finetuning_mode, num_batches=10, batch_size=32):
    """Return the correctly-formatted cov_dir path for a given model and ft mode."""
    return (
        f"results/{model}/covariances_strain"
        f"_n{num_batches}_b{batch_size}_tsm_attnsplit_efull_ft{finetuning_mode}"
    )


def fix_cov_dir(record):
    """Patch cov_dir to the canonical path if it looks malformed.

    The bash expansion bug produced paths like 'covariances_strain_n<ftmode>'
    instead of 'covariances_strain_n10_b32_tsm_attnsplit_efull_ft<ftmode>'.
    Skipped for regmean, which may use a custom cov_dir.
    """
    if record.get("merge_func") == "regmean":
        return False
    model = record.get("model")
    ft_mode = record.get("finetuning_mode")
    if not model or not ft_mode:
        return False
    correct = canonical_cov_dir(model, ft_mode)
    if record.get("cov_dir") != correct:
        record["cov_dir"] = correct
        return True
    return False


def recompute_hash(record):
    script = record.get("script", "eval_task_addition")
    args_dict = record_to_args_dict(record)
    # Remove 'script' itself — make_run_hash prepends it separately.
    args_dict.pop("script", None)
    ns = _FakeNamespace(args_dict)
    return make_run_hash(script, ns, ignore=_HASH_IGNORE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", default="results/results.jsonl", help="Path to the JSONL file."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print changes without writing."
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Drop the later copy of any duplicate records (same hash after rehashing).",
    )
    parser.add_argument(
        "--fix-cov-dir",
        action="store_true",
        help="Patch cov_dir to the canonical path (fixes bash variable expansion bug). "
        "Applied before rehashing. Skipped for regmean records.",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"File not found: {db_path}")
        return

    records = []
    with open(db_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    cov_fixed = 0
    changed = 0
    seen_hashes = {}  # new_hash -> first record index
    duplicates = []

    for i, record in enumerate(records):
        if args.fix_cov_dir and fix_cov_dir(record):
            print(
                f"[{i}] cov_dir fixed | model={record.get('model')} "
                f"| finetuning_mode={record.get('finetuning_mode')} "
                f"| merge_func={record.get('merge_func')}"
            )
            print(f"      cov_dir -> {record['cov_dir']}")
            cov_fixed += 1

        old_hash = record.get("run_hash")
        new_hash = recompute_hash(record)

        if old_hash != new_hash:
            print(
                f"[{i}] {record.get('script')} | model={record.get('model')} "
                f"| merge_func={record.get('merge_func')} | finetuning_mode={record.get('finetuning_mode')}"
            )
            print(f"      hash: {old_hash} -> {new_hash}")
            record["run_hash"] = new_hash
            changed += 1

        if new_hash in seen_hashes:
            duplicates.append((seen_hashes[new_hash], i, new_hash))
        else:
            seen_hashes[new_hash] = i

    if args.fix_cov_dir:
        print(f"\n{cov_fixed}/{len(records)} records had cov_dir patched.")
    print(f"{changed}/{len(records)} records had their hash updated.")

    duplicate_indices = {second for _, second, _ in duplicates}

    if duplicates:
        print(f"\nWARNING: {len(duplicates)} duplicate(s) detected after rehashing:")
        for first, second, h in duplicates:
            r1, r2 = records[first], records[second]
            action = "will drop" if args.deduplicate else "keeping both —"
            print(
                f"  hash={h}  records [{first}] (ts={r1.get('timestamp')}) "
                f"and [{second}] (ts={r2.get('timestamp')})  <- {action} [{second}]"
            )
        if not args.deduplicate:
            print("  Re-run with --deduplicate to drop the later copy of each pair.")

    if args.dry_run:
        print("\nDry-run mode — no files written.")
        return

    # Back up original before writing.
    backup = db_path.with_suffix(".jsonl.bak")
    shutil.copy2(db_path, backup)
    print(f"\nBackup written to {backup}")

    output = [
        r
        for i, r in enumerate(records)
        if not (args.deduplicate and i in duplicate_indices)
    ]
    with open(db_path, "w") as f:
        for record in output:
            f.write(json.dumps(record) + "\n")

    dropped = len(records) - len(output)
    print(
        f"Rewrote {len(output)} records to {db_path}"
        + (f" ({dropped} duplicate(s) dropped)" if dropped else "")
    )


if __name__ == "__main__":
    main()
