import json, os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── 1. Configuration (Restored from previous context) ───────────────────
BASE_FONTSIZE = 36  # As requested for high-res visibility
sns.set_theme(
    style="ticks",
    rc={
        "font.size": BASE_FONTSIZE,
        "axes.titlesize": BASE_FONTSIZE,
        "axes.labelsize": BASE_FONTSIZE,
        "xtick.labelsize": BASE_FONTSIZE * 0.9,
        "ytick.labelsize": BASE_FONTSIZE * 0.9,
        "legend.fontsize": BASE_FONTSIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "hatch.linewidth": 2.5,
        "hatch.color": "#333333",
    },
)

MODELS = {
    "Language": {"t5-base": "T5-Base", "t5-large": "T5-Large"},
    "Vision": {"ViT-B-16": "ViT-B/16", "ViT-B-32": "ViT-B/32", "ViT-L-14": "ViT-L/14"},
}

COLORS = {
    "EigenCov": "#00A658",
    "TSV": "#4A5568",
    "KNOTS-TSV": "#4A5568",
    "ISO-C": "#718096",
    "KNOTS-ISO-C": "#718096",
    "Average": "#A0AEC0",
    "RegMean": "#CBD5E0",
    "TA": "#E2E8F0",
}

BAR_ORDER = ["TA", "RegMean", "Average", "ISO-C", "TSV", "EigenCov"]
KNOTS_MAP = {"KNOTS-TSV": "TSV", "KNOTS-ISO-C": "ISO-C"}
DATA_NEEDING = ["TA", "RegMean", "KNOTS-TSV", "KNOTS-ISO-C"]
TEXTURE_SLASH = "//"
TEXTURE_DOT = ".."

# ── 2. Data Loading (Assuming standard format) ─────────────────────────
# Re-including the processing logic to ensure 'df' and 'METHODS' map exist
METHODS_MAP = {
    "sum_data": "TA",
    "regmean": "RegMean",
    "mean": "Average",
    "isoc": "ISO-C",
    "isoc_mean": "ISO-C",
    "tsv": "TSV",
    "eigcov": "EigenCov",
    "knots_tsv": "KNOTS-TSV",
    "knots_isoc_mean": "KNOTS-ISO-C",
    "expert": "Expert",
    "zeroshot": "Zeroshot",
}


def load_data():
    paths = [
        "results-tracked/results.jsonl",
        "results-tracked/results-anc.jsonl",
        "results-tracked/results-ta.jsonl",
    ]
    all_data = []
    for p in paths:
        if osp.exists(p):
            with open(p) as f:
                all_data.extend([json.loads(l) for l in f if l.strip()])

    mapping = {k: v for d in MODELS.values() for k, v in d.items()}
    return (
        pd.DataFrame(all_data)
        .assign(
            Model=lambda d: d["model"].map(mapping),
            Method=lambda d: d["merge_func"].map(METHODS_MAP),
            Score=lambda d: d.get("test_avg_topk", d["test_avg_top1"]).fillna(
                d["test_avg_top1"]
            )
            * 100,
        )
        .dropna(subset=["Model", "Method"])
    )


df = load_data()

# ── 2. Plotting Logic (2x2 Grid) ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(32, 24))
legend_data = {}

# Grid Structure:
# Row 0: Language (Std, LoRA)
# Row 1: Vision (Std, LoRA)
grid_config = [
    (0, 0, "standard", "Language"),
    (0, 1, "lora", "Language"),
    (1, 0, "standard", "Vision"),
    (1, 1, "lora", "Vision"),
]

for r, c, mode, dom in grid_config:
    ax = axes[r, c]
    models = list(MODELS[dom].values())
    sub = df[df["finetuning_mode"] == mode]
    tbl = sub.groupby(["Method", "Model"])["Score"].max().unstack("Model")

    x, bw = np.arange(len(models)), 0.14
    gw = len(BAR_ORDER) * bw

    for j, m in enumerate(BAR_ORDER):
        if m not in tbl.index:
            continue
        vals = [tbl.loc[m, mod] if mod in tbl.columns else 0 for mod in models]

        hatch = TEXTURE_SLASH if m in DATA_NEEDING else None
        rects = ax.bar(
            x - gw / 2 + (j + 0.5) * bw,
            vals,
            width=bw,
            color=COLORS.get(m),
            edgecolor="white",
            linewidth=1.5,
            hatch=hatch,
            zorder=3,
        )

        # Capture legend handles
        legend_data[m] = rects

        # Handle KNOTS Gains (LoRA columns only)
        knots_key = next((k for k, v in KNOTS_MAP.items() if v == m), None)
        if mode == "lora" and knots_key in tbl.index:
            k_vals = [
                tbl.loc[knots_key, mod] if mod in tbl.columns else 0 for mod in models
            ]
            gains = [max(k - b, 0) for k, b in zip(k_vals, vals)]

            k_rects = ax.bar(
                x - gw / 2 + (j + 0.5) * bw,
                gains,
                bottom=vals,
                width=bw,
                color=COLORS.get(m),
                edgecolor="white",
                hatch=TEXTURE_DOT,
                zorder=4,
            )
            legend_data[knots_key] = k_rects

    # ── Honest Scaling (45-85 for Lang, 45-95 for Vis) ──
    ymin, ymax = (45, 85) if dom == "Language" else (45, 95)
    ax.set_ylim(ymin, ymax)

    # Baselines
    for bl, style in {"Expert": "--", "Zeroshot": ":"}.items():
        if bl in tbl.index:
            for k, mod in enumerate(models):
                if mod in tbl.columns:
                    line = ax.hlines(
                        tbl.loc[bl, mod],
                        k - gw / 2 - 0.05,
                        k + gw / 2 + 0.05,
                        colors="black" if bl == "Expert" else "gray",
                        linestyles=style,
                        lw=3.5,
                        zorder=5,
                    )
                    legend_data[bl] = line

    # Formatting
    title_mode = "Standard" if mode == "standard" else "LoRA"
    ax.set_title(f"{dom} ({title_mode})", fontweight="bold", pad=30)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.3, ls="--")

    # Y-label only on the left-most plots
    if c == 0:
        ax.set_ylabel("Accuracy (%)", fontweight="bold", labelpad=20)

# ── 3. Global Legend & Export ──────────────────────────────────────────
order = [
    "Expert",
    "Zeroshot",
    "TA",
    "RegMean",
    "Average",
    "ISO-C",
    "KNOTS-ISO-C",
    "TSV",
    "KNOTS-TSV",
    "EigenCov",
]
legend_data["Expert"] = plt.Line2D([0], [0], color="black", ls="--", lw=3.5)
legend_data["Zeroshot"] = plt.Line2D([0], [0], color="gray", ls=":", lw=3.5)

final_handles = [legend_data[m] for m in order if m in legend_data]
fig.legend(
    final_handles,
    [m for m in order if m in legend_data],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.04),
    ncol=5,
    frameon=False,
    columnspacing=1.5,
)

plt.tight_layout()
# Adjust bottom to make room for the massive legend
plt.subplots_adjust(bottom=0.15, hspace=0.3, wspace=0.2)
plt.savefig("results-tracked/performance-grid.pdf", dpi=300, bbox_inches="tight")
plt.show()
