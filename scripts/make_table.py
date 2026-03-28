import json, pandas as pd
import os.path as osp

FT_MODE = "standard"
RESULTS = osp.join("../results-tracked", "results.jsonl")
MODELS = {
    "t5-base": "T5-Base",
    "t5-large": "T5-Large",
    "Olmo-3-7B": "Olmo",
    "ViT-B-16": "ViT-B/16",
    "ViT-B-32": "ViT-B/32",
    "ViT-L-14": "ViT-L/14",
}
METHODS = {
    # ==== Standard methods ====
    "mean": "Average",
    "sum": "TA",
    "regmean": "RegMean",
    "eigcov": "EigenCov",
    "tsv": "TSV",
    "isoc": "ISO-C",
    # # ==== LoRA-only methods ====
    # "mean": "Average",
    # "sum": "TA",
    # "regmean": "RegMean",
    # "eigcov": "EigenCov",
    # "tsv": "TSV",
    # "isoc_mean": "ISO-C",
    # "knots_tsv": "KNOTS-TSV",
    # "knots_isoc_mean": "KNOTS-ISO-C",
}

DATA_FREE = {
    "RegMean": False,
    "EigenCov": True,
    "TSV": True,
    "ISO-C": True,
    "KNOTS-TSV": True,
    "KNOTS-ISO-C": True,
}

with open(RESULTS) as f:
    df = pd.DataFrame(json.loads(l) for l in f if l.strip())

df["Model"] = df["model"].map(MODELS)
df["Method"] = df["merge_func"].map(METHODS)
df = df[df["finetuning_mode"] == FT_MODE]
df["score"] = df["test_avg_top1"]
df = df.dropna(subset=["Model", "Method", "score"])

# print(df.to_string())

tbl = (
    df.groupby(["Method", "Model"])["score"]
    .max()
    .unstack("Model")
    .reindex(index=METHODS.values(), columns=MODELS.values())
    .map(lambda x: "" if pd.isna(x) else f"{100*x:.1f}")
    .rename_axis(index=None, columns=None)
)

# insert a column for data-free
tbl.insert(
    0,
    "Data-free",
    [r"\cmark" if DATA_FREE.get(m, False) else r"\xmark" for m in tbl.index],
)

print(tbl.to_latex(escape=False, column_format="l" + "c" * (len(MODELS) + 1)))
