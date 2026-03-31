import json, os.path as osp
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
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
        'https://github.com/rsms/inter/releases/download/v3.19/Inter-3.19.zip',
        'Inter-3.19.zip'
    )
    with zipfile.ZipFile('Inter-3.19.zip', 'r') as zip_ref:
        zip_ref.extractall()
    matplotlib.font_manager.fontManager.addfont('Inter Variable/Inter.ttf')
    matplotlib.rcParams.update({"font.family": "Inter"})
except:
    pass  # Fallback to default if Inter font unavailable

# ── 1. Configuration ────────────────────────────────────────────────────
BASE_FONTSIZE = 12  # Display-friendly size

MODELS = {
    "Language": {"t5-base": "T5-Base", "t5-large": "T5-Large"},
    "Vision": {"ViT-B-16": "ViT-B/16", "ViT-B-32": "ViT-B/32", "ViT-L-14": "ViT-L/14"},
}

# Vibrant but controlled palette so methods are distinct without feeling loud.
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
KNOTS_MAP = {"KNOTS-TSV": "TSV", "KNOTS-ISO-C": "ISO-C"}
DATA_NEEDING = ["TA", "RegMean"]
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
    "eigcov": "ACTMat",
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

# ── 2. Plotting Logic (1x4 Grid) ───────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
legend_data = {}

grid_config = [
    ("standard", "Language"),
    ("lora", "Language"),
    ("standard", "Vision"),
    ("lora", "Vision"),
]

for idx, (mode, dom) in enumerate(grid_config):
    ax = axes[idx]
    models = list(MODELS[dom].values())
    sub = df[df["finetuning_mode"] == mode]
    tbl = sub.groupby(["Method", "Model"])["Score"].max().unstack("Model")

    x = np.arange(len(models)) if mode == "lora" else np.arange(len(models)) * 1.25
    # Standard uses extra spacing between methods; LoRA stays tighter to fit more bars.
    bw = 0.09 if mode == "lora" else 0.13
    group_step = bw if mode == "lora" else bw * 1.12

    # Filter bar order based on mode and center each series in the group.
    bar_order_filtered = (
        BAR_ORDER if mode == "lora" else [m for m in BAR_ORDER if not m.startswith("KNOTS")]
    )
    offsets = (
        np.arange(len(bar_order_filtered)) - (len(bar_order_filtered) - 1) / 2
    ) * group_step
    gw = len(bar_order_filtered) * group_step

    for j, m in enumerate(bar_order_filtered):
        if m not in tbl.index:
            continue
        vals = [tbl.loc[m, mod] if mod in tbl.columns else 0 for mod in models]

        rects = ax.bar(
            x + offsets[j],
            vals,
            width=bw,
            color=COLORS.get(m),
            edgecolor="white",
            linewidth=1.5,
            hatch=TEXTURE_SLASH if m in DATA_NEEDING else None,
            alpha=1.0 if m == "ACTMat" else 0.6,
            zorder=3,
        )

        # Capture legend handles
        legend_data[m] = rects

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
                        x[k] - gw / 2 - 0.05,
                        x[k] + gw / 2 + 0.05,
                        colors="black" if bl == "Expert" else "gray",
                        linestyles=style,
                        lw=3.5,
                        zorder=5,
                    )
                    legend_data[bl] = line

    # Formatting
    title_mode = "Full-Model Fine-Tuning" if mode == "standard" else "LoRA"
    ax.set_title(f"{dom} ({title_mode})", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.tick_params(axis="x", pad=10)
    ax.yaxis.grid(True, alpha=0.3, ls="--")

    # Y-label only on the left-most plots
    if idx == 0:
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
    "ACTMat",
]
legend_data["Expert"] = plt.Line2D([0], [0], color="black", ls="--", lw=3.5)
legend_data["Zeroshot"] = plt.Line2D([0], [0], color="gray", ls=":", lw=3.5)

final_handles = [legend_data[m] for m in order if m in legend_data]

plt.tight_layout()
# Adjust spacing to prevent title/label overlap
plt.subplots_adjust(bottom=0.27, top=0.88, wspace=0.3)

note_ax = fig.add_axes([0.04, 0.0, 0.28, 0.18])
note_ax.axis("off")
note_ax.text(
    0.0,
    0.5,
    "Hatched bars indicate methods that depend on training data.",
    ha="left",
    va="center",
    fontsize=20,
    color="#4A5568",
)

legend_ax = fig.add_axes([0.34, 0.0, 0.62, 0.18])
legend_ax.axis("off")
legend_ax.legend(
    final_handles,
    [m for m in order if m in legend_data],
    loc="center",
    ncol=5,
    frameon=False,
    columnspacing=1.5,
)

sns.despine()
plt.savefig("results-tracked/performance-grid.pdf", dpi=300, bbox_inches="tight")
plt.show()
