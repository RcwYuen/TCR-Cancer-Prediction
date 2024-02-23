import pandas as pd
from pathlib import Path
import os

df = pd.read_csv("tcrseq_pbmcfnames.csv")
dups = df["LTX_ID"].value_counts().index[(df["LTX_ID"].value_counts() > 2).values]

to_discard = []

for i in dups:
    repeats = df.loc[df["LTX_ID"] == i]["filename"].tolist()
    to_discard += [i for i in repeats if "merge" not in i]

directory = Path.cwd().parent / "data" / "full-trimmed" / "pbmc_cancer"
for i in to_discard:
    fname = i.replace(".gz", "")
    try:
        os.remove(directory / fname)
    except FileNotFoundError as e:
        print (f"{directory / fname} not found")
