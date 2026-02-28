import json, pandas as pd

df = pd.DataFrame(json.loads(l) for l in open("../results/results.jsonl") if l.strip())

COLS = ["T5-Base", "T5-Large", "ViT-B/16", "ViT-B/32", "ViT-L/14"]
model_map = {"ViT-B-16": "ViT-B/16", "ViT-B-32": "ViT-B/32", "ViT-L-14": "ViT-L/14"}
method_map = {
    "regmean": "RegMean",
    "eigcov": "EigenCov",
    "tsv": "TSV",
    "isoc_mean": "ISO-C",
    "knots_tsv": "KNOTS-TSV",
    "knots_isoc_mean": "KNOTS-ISO-C",
}
data_free = {"RegMean": True}  # fill as you like

df["model"] = df["model"].replace(model_map)
df["method"] = df["merge_func"].replace(method_map)

tbl = (
    df.groupby(["method", "model"])["test_avg_top1"]
    .max()
    .unstack("model")
    .reindex(columns=COLS)
)

tbl.insert(
    0,
    "Data-free",
    [r"\xmark" if data_free.get(m, False) else r"\cmark" for m in tbl.index],
)
tbl.insert(0, "Method", [rf"\textsc{{{m}}}" for m in tbl.index])
tbl = tbl.applymap(lambda x: "" if pd.isna(x) else f"{x:.1f}")

print(tbl.reset_index(drop=True).to_latex(index=False, escape=False))
