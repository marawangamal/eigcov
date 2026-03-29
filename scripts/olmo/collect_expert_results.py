"""Collect olmes evaluation results and compute domain averages.

Usage:
    python scripts/nlg/collect_results.py --dirs results-nlg-4096-eigcov results-nlg-4096-mean
"""

import argparse
import json
import sys
from pathlib import Path

from src.results_db import append_result, args_to_dict, make_run_hash, record_exists

STANDALONE = {
    "codex_humaneval::tulu": "HumanEval",
    "codex_humanevalplus::tulu": "HumanEval+",
    "ifeval::tulu": "IFEval",
    "aime:zs_cot_r1::pass_at_32_2024_deepseek": "AIME 2024",
    "aime:zs_cot_r1::pass_at_32_2025_deepseek": "AIME 2025",
}

EXPERT_BENCHMARKS = {
    "Math": ["AIME 2024", "AIME 2025"],
    "Code": ["HumanEval", "HumanEval+"],
    "IF": ["IFEval"],
}

DISPLAY_ORDER = ["HumanEval", "HumanEval+", "IFEval", "AIME 2024", "AIME 2025"]
SCORE_CONFIGS = {
    "@1": {
        "codex_humaneval::tulu": "pass_at_1",
        "codex_humanevalplus::tulu": "pass_at_1",
        "aime:zs_cot_r1::pass_at_32_2024_deepseek": "pass_at_1",
        "aime:zs_cot_r1::pass_at_32_2025_deepseek": "pass_at_1",
    },
    "@k": {
        "codex_humaneval::tulu": "pass_at_10",
        "codex_humanevalplus::tulu": "pass_at_10",
        "aime:zs_cot_r1::pass_at_32_2024_deepseek": "pass_at_32",
        "aime:zs_cot_r1::pass_at_32_2025_deepseek": "pass_at_32",
    },
}


def load_results(results_dir: Path, primary_scores: dict[str, str]) -> dict[str, float]:
    metrics_files = sorted(results_dir.glob("*-metrics.json"))
    if not metrics_files:
        print(f"Warning: no metrics files found in {results_dir}", file=sys.stderr)
        return {}

    task_scores: dict[str, float] = {}
    for f in metrics_files:
        data = json.loads(f.read_text())
        task_name = data["task_config"]["metadata"]["alias"]
        primary_key = primary_scores.get(task_name, "primary_score")
        primary = data.get("metrics", {}).get(primary_key)
        if primary is not None:
            task_scores[task_name] = primary

    scores: dict[str, float] = {}

    for task_name, display_name in STANDALONE.items():
        if task_name in task_scores:
            scores[display_name] = task_scores[task_name]

    return scores


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect olmes results and compute averages."
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="One or more olmes output directories containing *-metrics.json files.",
    )
    parser.add_argument(
        "--no-code",
        action="store_true",
        help="Omit code benchmarks (HumanEval, HumanEval+) from the table.",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Append average accuracies to results/results.jsonl.",
    )
    parser.add_argument(
        "--results-db",
        default="results-tracked/results.jsonl",
        help="Path to the results JSONL database (default: results/results.jsonl).",
    )
    return parser.parse_args()


def print_table(all_results: dict[str, dict[str, float]], label: str):
    methods = list(all_results.keys())
    col_w = max(15, *(len(m) + 2 for m in methods))

    print(f"\n=== {label} ===")
    header = f"{'Benchmark':<15}" + "".join(f"{m:>{col_w}}" for m in methods)
    print(header)
    print("-" * len(header))

    for bench in DISPLAY_ORDER:
        row = f"{bench:<15}"
        for m in methods:
            val = all_results[m].get(bench)
            row += f"{val:{col_w}.3f}" if val is not None else f"{'—':>{col_w}}"
        print(row)

    print("-" * len(header))
    row = f"{'Average':<15}"
    averages: dict[str, float] = {}
    for m in methods:
        vals = [all_results[m][b] for b in DISPLAY_ORDER if b in all_results[m]]
        if vals:
            averages[m] = sum(vals) / len(vals)
            row += f"{averages[m]:{col_w}.3f}"
        else:
            row += f"{'—':>{col_w}}"
    print(row)
    return averages


def main():
    args = parse_args()

    # Collect scores for each config (pass@1, pass@k)
    # results_by_config[suffix]["expert"] = {benchmark: score}
    results_by_config: dict[str, dict[str, dict[str, float]]] = {}
    for suffix, primary_scores in SCORE_CONFIGS.items():
        expert: dict[str, float] = {}
        for d in args.dirs:
            p = Path(d)
            expert_type = p.name.rsplit("-", 1)[-1]
            benchmarks = EXPERT_BENCHMARKS.get(expert_type)
            if benchmarks is None:
                print(
                    f"Warning: unknown expert type '{expert_type}' from {p.name}",
                    file=sys.stderr,
                )
                continue
            scores = load_results(p, primary_scores)
            for bench in benchmarks:
                if bench in scores:
                    expert[bench] = scores[bench]
        results_by_config[suffix] = {"expert": expert}

    # Print tables and collect averages per config
    averages_by_config: dict[str, dict[str, float]] = {}
    for suffix, results in results_by_config.items():
        averages_by_config[suffix] = print_table(results, f"pass{suffix}")

    if args.log:
        _HASH_IGNORE = {"log", "no_code", "results_db"}
        args.merge_func = "expert"
        run_hash = make_run_hash("collect_results_olmo", args, ignore=_HASH_IGNORE)
        if record_exists(args.results_db, run_hash):
            print("Skipping expert: already logged")
            return

        record = {
            **args_to_dict(args),
            "script": "collect_results_olmo",
            "model": "Olmo-3-7B",
            "finetuning_mode": "standard",
        }

        # Add prefixed scores for each config
        for suffix, results in results_by_config.items():
            for bench, score in results["expert"].items():
                record[f"test_{bench}{suffix}"] = score
            avg = averages_by_config[suffix].get("expert")
            if avg is not None:
                record[f"avg_accuracy{suffix}"] = avg

        # test_avg_top1 = avg_accuracy@1 for compatibility with make_table.py
        if "@1" in averages_by_config and "expert" in averages_by_config["@1"]:
            record["test_avg_top1"] = averages_by_config["@1"]["expert"]
        print("Logging record with merge_func:", record["merge_func"])
        append_result(args.results_db, record, run_hash)
        print(f"\nLogged 1 result(s) to {args.results_db}")


if __name__ == "__main__":
    main()
