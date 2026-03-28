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

DISPLAY_ORDER = ["HumanEval", "HumanEval+", "IFEval", "AIME 2024", "AIME 2025"]
PRIMARY_SCORES = {
    "codex_humaneval::tulu": "pass_at_1",
    "codex_humanevalplus::tulu": "pass_at_1",
    "aime:zs_cot_r1::pass_at_32_2024_deepseek": "pass_at_1",
}

# PRIMARY_SCORES = {
#     "codex_humaneval::tulu": "pass_at_10",
#     "codex_humanevalplus::tulu": "pass_at_10",
#     "aime:zs_cot_r1::pass_at_32_2024_deepseek": "pass_at_32",
# }


def load_results(results_dir: Path) -> dict[str, float]:
    metrics_files = sorted(results_dir.glob("*-metrics.json"))
    if not metrics_files:
        print(f"Warning: no metrics files found in {results_dir}", file=sys.stderr)
        return {}

    task_scores: dict[str, float] = {}
    for f in metrics_files:
        data = json.loads(f.read_text())
        task_name = data["task_config"]["metadata"]["alias"]
        primary_key = PRIMARY_SCORES.get(task_name, "primary_score")
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


def main():
    args = parse_args()

    all_results: dict[str, dict[str, float]] = {}
    for d in args.dirs:
        p = Path(d)
        name = p.name.replace("results-nlg-", "")
        all_results[name] = load_results(p)

    display_order = [b for b in DISPLAY_ORDER]

    methods = list(all_results.keys())
    col_w = max(15, *(len(m) + 2 for m in methods))

    header = f"{'Benchmark':<15}" + "".join(f"{m:>{col_w}}" for m in methods)
    print(header)
    print("-" * len(header))

    for bench in display_order:
        row = f"{bench:<15}"
        for m in methods:
            val = all_results[m].get(bench)
            row += f"{val:{col_w}.3f}" if val is not None else f"{'—':>{col_w}}"
        print(row)

    print("-" * len(header))
    row = f"{'Average':<15}"
    averages: dict[str, float] = {}
    for m in methods:
        vals = [all_results[m][b] for b in display_order if b in all_results[m]]
        if vals:
            averages[m] = sum(vals) / len(vals)
            row += f"{averages[m]:{col_w}.3f}"
        else:
            row += f"{'—':>{col_w}}"
    print(row)

    if args.log and averages:
        _HASH_IGNORE = {"log", "no_code", "results_db"}
        logged = 0
        for method, avg in averages.items():
            args.merge_func = method
            run_hash = make_run_hash("collect_results_olmo", args, ignore=_HASH_IGNORE)
            if record_exists(args.results_db, run_hash):
                print(f"Skipping {method}: already logged")
                continue
            record = {
                "script": "collect_results_olmo",
                "model": "Olmo-3-7B",
                "merge_func": method,
                **args_to_dict(args),
                "avg_accuracy": avg,
                **{f"test_{k}": v for k, v in all_results[method].items()},
            }
            append_result(args.results_db, record, run_hash)
            logged += 1
        print(f"\nLogged {logged} result(s) to {args.results_db}")


if __name__ == "__main__":
    main()
