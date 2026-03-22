"""Collect olmes evaluation results and compute domain averages.

Usage:
    python scripts/nlg/collect_results.py --dirs results-nlg-4096-eigcov results-nlg-4096-mean
"""

import argparse
import json
import sys
from pathlib import Path

GROUPS = {
    "bbh": "BBH-CoT",
    "minerva_math": "MATH",
}

STANDALONE = {
    "codex_humaneval": "HumanEval",
    "codex_humanevalplus": "HumanEval+",
    "drop": "DROP",
    "gsm8k": "GSM8K",
    "ifeval": "IFEval",
    "popqa": "PopQA",
}

DISPLAY_ORDER = [
    "BBH-CoT",
    "HumanEval",
    "HumanEval+",
    "DROP",
    "GSM8K",
    "IFEval",
    "MATH",
    "PopQA",
]


def load_results(results_dir: Path) -> dict[str, float]:
    metrics_files = sorted(results_dir.glob("*-metrics.json"))
    if not metrics_files:
        print(f"Warning: no metrics files found in {results_dir}", file=sys.stderr)
        return {}

    task_scores: dict[str, float] = {}
    for f in metrics_files:
        data = json.loads(f.read_text())
        task_name = data.get("task_name", "")
        primary = data.get("metrics", {}).get("primary_score")
        if primary is not None:
            task_scores[task_name] = primary

    scores: dict[str, float] = {}

    for prefix, display_name in GROUPS.items():
        subtask_scores = [v for k, v in task_scores.items() if k.startswith(prefix)]
        if subtask_scores:
            scores[display_name] = sum(subtask_scores) / len(subtask_scores)

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
    return parser.parse_args()


def main():
    args = parse_args()

    all_results: dict[str, dict[str, float]] = {}
    for d in args.dirs:
        p = Path(d)
        name = p.name.replace("results-nlg-", "")
        all_results[name] = load_results(p)

    methods = list(all_results.keys())
    col_w = max(15, *(len(m) + 2 for m in methods))

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
    for m in methods:
        vals = [all_results[m][b] for b in DISPLAY_ORDER if b in all_results[m]]
        row += f"{sum(vals)/len(vals):{col_w}.3f}" if vals else f"{'—':>{col_w}}"
    print(row)


if __name__ == "__main__":
    main()
