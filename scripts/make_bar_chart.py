import json, os.path as osp
import matplotlib.pyplot as plt, numpy as np, pandas as pd

# ── Params (toggle here) ────────────────────────────────────────────────
# FT_MODE = "standard"  # "standard" | "lora"
FT_MODE = "lora"

# ── Config ──────────────────────────────────────────────────────────────
RESULTS = osp.join("../results-tracked", "results.jsonl")
RESULTS_ANC = osp.join("../results-tracked", "results-anc.jsonl")

LANG_MODELS = {"t5-base": "T5-Base", "t5-large": "T5-Large"}
VIS_MODELS = {"ViT-B-16": "ViT-B/16", "ViT-B-32": "ViT-B/32", "ViT-L-14": "ViT-L/14"}
ALL_MODELS = {**LANG_MODELS, **VIS_MODELS}

# # ==== Standard ====
# BAR_METHODS = {
#     "sum": "TA", "regmean": "RegMean", "mean": "Average",
#     "isoc": "ISO-C", "tsv": "TSV", "eigcov": "EigenCov",
# }
# KNOTS_METHODS = {}
# KNOTS_BASE = {}

# ==== LoRA ====
BAR_METHODS = {
    "sum": "TA",
    "regmean": "RegMean",
    "mean": "Average",
    "isoc_mean": "ISO-C",
    "tsv": "TSV",
    "eigcov": "EigenCov",
}
KNOTS_METHODS = {"knots_tsv": "KNOTS-TSV", "knots_isoc_mean": "KNOTS-ISO-C"}
KNOTS_BASE = {"KNOTS-TSV": "TSV", "KNOTS-ISO-C": "ISO-C"}

BASELINE_METHODS = {"expert": "Expert", "zeroshot": "Zeroshot"}
ALL_METHODS = {**BAR_METHODS, **KNOTS_METHODS, **BASELINE_METHODS}

DATA_NEEDED = {"RegMean", "TA"}
COLORS = {
    "EigenCov": "#00A658",
    "TSV": "#4A5568",
    "ISO-C": "#718096",
    "Average": "#A0AEC0",
    "RegMean": "#CBD5E0",
    "TA": "#E2E8F0",
}
BL_STYLES = {
    "Expert": dict(color="black", ls="--", lw=1.2),
    "Zeroshot": dict(color="gray", ls=":", lw=1.2),
}


# ── Load data ───────────────────────────────────────────────────────────
def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


df = pd.DataFrame(_read_jsonl(RESULTS))

if osp.exists(RESULTS_ANC):
    df = pd.concat([df, pd.DataFrame(_read_jsonl(RESULTS_ANC))])

df = (
    df[df["finetuning_mode"] == FT_MODE]
    .assign(
        Model=lambda d: d["model"].map(ALL_MODELS),
        Method=lambda d: d["merge_func"].map(ALL_METHODS),
        score=lambda d: d.get("test_avg_topk", d["test_avg_top1"]).fillna(
            d["test_avg_top1"]
        ),
    )
    .dropna(subset=["Model", "Method", "score"])
)

tbl = df.groupby(["Method", "Model"])["score"].max().unstack("Model")

# ── Plot: two subplots (Language | Vision) ──────────────────────────────
panels = [
    ("Language", [v for v in LANG_MODELS.values() if v in tbl.columns]),
    ("Vision", [v for v in VIS_MODELS.values() if v in tbl.columns]),
]
bars = [v for v in BAR_METHODS.values() if v in tbl.index]
baselines = [v for v in BASELINE_METHODS.values() if v in tbl.index]
n_bars = len(bars)
bw = 0.12
gw = n_bars * bw

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

for ax, (title, models) in zip(axes, panels):
    x = np.arange(len(models))

    for i, m in enumerate(bars):
        vals = [
            tbl.loc[m, mod] * 100 if pd.notna(tbl.loc[m, mod]) else 0 for mod in models
        ]
        hatch = "///" if m in DATA_NEEDED else None
        ax.bar(
            x - gw / 2 + (i + 0.5) * bw,
            vals,
            width=bw,
            color=COLORS.get(m, "#999"),
            edgecolor="white",
            linewidth=0.8,
            hatch=hatch,
            zorder=3,
            label=m if ax is axes[0] else None,
        )

    # KNOTS stacked segments on top of their base bars
    for knots_name, base_name in KNOTS_BASE.items():
        if knots_name not in tbl.index or base_name not in bars:
            continue
        i = bars.index(base_name)
        base_vals = [
            tbl.loc[base_name, mod] * 100 if pd.notna(tbl.loc[base_name, mod]) else 0
            for mod in models
        ]
        knots_vals = [
            tbl.loc[knots_name, mod] * 100 if pd.notna(tbl.loc[knots_name, mod]) else 0
            for mod in models
        ]
        gains = [max(k - b, 0) for k, b in zip(knots_vals, base_vals)]
        ax.bar(
            x - gw / 2 + (i + 0.5) * bw,
            gains,
            bottom=base_vals,
            width=bw,
            color=COLORS.get(base_name, "#999"),
            edgecolor="white",
            linewidth=0.8,
            hatch="...",
            alpha=0.7,
            zorder=3,
            label=knots_name if ax is axes[0] else None,
        )

    for bl in baselines:
        for j, mod in enumerate(models):
            if bl in tbl.index and mod in tbl.columns and pd.notna(tbl.loc[bl, mod]):
                val = tbl.loc[bl, mod] * 100
                ax.hlines(
                    val,
                    j - gw / 2 - bw * 0.3,
                    j + gw / 2 + bw * 0.3,
                    zorder=4,
                    **BL_STYLES[bl],
                    label=bl if ax is axes[0] and j == 0 else None,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(45 if title == "Vision" else 40, 95)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.4, linestyle="--", zorder=0)

axes[0].set_ylabel("Accuracy (%)", fontsize=12)

# Shared legend at bottom
h, l = axes[0].get_legend_handles_labels()
by_label = dict(zip(l, h))
fig.legend(
    by_label.values(),
    by_label.keys(),
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=n_bars + len(KNOTS_BASE) + len(baselines),
    fontsize=9,
    frameon=False,
)
fig.tight_layout()
fig.subplots_adjust(bottom=0.12)
plt.show()
