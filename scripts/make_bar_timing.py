import json
import os.path as osp

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ── Font Setup ──────────────────────────────────────────────────────────
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("notebook", font_scale=1.75)

try:
    import urllib.request
    import zipfile

    urllib.request.urlretrieve(
        "https://github.com/rsms/inter/releases/download/v3.19/Inter-3.19.zip",
        "Inter-3.19.zip",
    )
    with zipfile.ZipFile("Inter-3.19.zip", "r") as zip_ref:
        zip_ref.extractall()
    matplotlib.font_manager.fontManager.addfont("Inter Variable/Inter.ttf")
    matplotlib.rcParams.update({"font.family": "Inter"})
except Exception:
    pass  # Fallback to default if Inter font unavailable

# ── Configuration ───────────────────────────────────────────────────────
COLORS = {
    "TA": "#5B8FF9",
    "RegMean": "#61DDAA",
    "Average": "#9C88FF",
    "ISO-C": "#F6BD16",
    "TSV": "#5D7092",
    "ACTMat": "#E8684A",
    "KNOTS-ISO-C": "#FF9D4D",
    "KNOTS-TSV": "#6DC8EC",
}

BAR_ORDER = ["TA", "RegMean", "Average", "ISO-C", "KNOTS-ISO-C", "TSV", "KNOTS-TSV", "ACTMat"]
DATA_NEEDING = ["TA", "RegMean"]
TEXTURE_SLASH = "//"
METHODS_MAP = {
    "sum": "TA",
    "sum_data": "TA",
    "regmean": "RegMean",
    "mean": "Average",
    "isoc": "ISO-C",
    "isoc_mean": "ISO-C",
    "tsv": "TSV",
    "eigcov": "ACTMat",
    "knots_tsv": "KNOTS-TSV",
    "knots_isoc_mean": "KNOTS-ISO-C",
}


def load_timing_data():
    path = "results-tracked/results-latency.jsonl"
    if not osp.exists(path):
        raise FileNotFoundError(f"Could not find timing results at {path}")

    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    return (
        pd.DataFrame(rows)
        .assign(
            Method=lambda d: d["merge_func"].map(METHODS_MAP),
            MergeTimeSeconds=lambda d: d["merge_time_seconds"],
        )
        .dropna(subset=["Method", "MergeTimeSeconds"])
    )


df = load_timing_data()
summary = (
    df.groupby("Method", as_index=False)["MergeTimeSeconds"]
    .mean()
    .assign(Method=lambda d: pd.Categorical(d["Method"], categories=BAR_ORDER, ordered=True))
    .sort_values("Method")
)

methods = summary["Method"].tolist()
times = summary["MergeTimeSeconds"].tolist()
x = range(len(methods))

fig, ax = plt.subplots(figsize=(11, 5.5))

for idx, (method, merge_time) in enumerate(zip(methods, times)):
    ax.bar(
        idx,
        merge_time,
        width=0.72,
        color=COLORS.get(method),
        edgecolor="white",
        linewidth=1.5,
        hatch=TEXTURE_SLASH if method in DATA_NEEDING else None,
        alpha=1.0 if method == "ACTMat" else 0.6,
        zorder=3,
    )

ax.set_xticks(list(x))
ax.set_xticklabels(methods, rotation=25, ha="right", fontweight="bold")
ax.set_ylabel("Merge Time (s)", fontweight="bold")
ax.set_title("Latency by Merge Method", fontweight="bold", pad=15)
ax.set_yscale("log")
ax.yaxis.grid(True, alpha=0.3, ls="--")
ax.set_axisbelow(True)

plt.tight_layout()
sns.despine()
plt.savefig("results-tracked/timing-bar.pdf", dpi=300, bbox_inches="tight")
plt.show()
