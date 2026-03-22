"""Collect olmes evaluation results and compute domain averages.

Usage:
    python scripts/nlg/collect_results.py --dir results-nlg-4096-eigcov
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
        "--dir",
        required=True,
        help="olmes output directory containing *-metrics.json files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.dir)
    scores = load_results(results_dir)

    print(f"{'Benchmark':<15} {'Score':>10}")
    print("-" * 26)

    for bench in DISPLAY_ORDER:
        val = scores.get(bench)
        if val is not None:
            print(f"{bench:<15} {val:>10.3f}")
        else:
            print(f"{bench:<15} {'—':>10}")

    print("-" * 26)
    vals = [scores[b] for b in DISPLAY_ORDER if b in scores]
    if vals:
        print(f"{'Average':<15} {sum(vals)/len(vals):>10.3f}")


if __name__ == "__main__":
    main()
