import json, pandas as pd
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results-tracked" / "results.jsonl"
MODELS = {
    "t5-base": "T5-Base",
    "t5-large": "T5-Large",
    "Olmo-3-7B": "OLMo-3-7B",
    "ViT-B-16": "ViT-B/16",
    "ViT-B-32": "ViT-B/32",
    "ViT-L-14": "ViT-L/14",
}
METHODS = {
    "mean": "Average",
    "sum": "TA",
    "regmean": "RegMean",
    "eigcov": "EigenCov",
    "tsv": "TSV",
    "isoc": "ISO-C",
    "eigcov_gd": "ECGD",
}

with open(RESULTS) as f:
    df = pd.DataFrame(json.loads(l) for l in f if l.strip())

df["Model"] = df["model"].map(MODELS)
df["Method"] = df["merge_func"].map(METHODS)
df["score"] = df["test_avg_top1"]
df = df.dropna(subset=["Model", "Method", "score"])

tbl = (
    df.groupby(["Method", "Model"])["score"]
    .max()
    .unstack("Model")
    .reindex(index=METHODS.values(), columns=MODELS.values())
    .map(lambda x: "" if pd.isna(x) else f"{100*x:.1f}")
    .rename_axis(index=None, columns=None)
)

print(tbl.to_latex(escape=False, column_format="l" + "c" * len(MODELS)))
