import pandas as pd
from pathlib import Path
import os
import shutil

def make_directory_where_necessary(directory):
    if not os.path.exists(directory):
        if make_directory_where_necessary(directory.parent):
            os.mkdir(directory)
    return True

def copy_files(src_path, dest_path):
    for file_path in src_path.iterdir():
        if file_path.is_file():
            dest_file_path = dest_path / file_path.name
            shutil.copy(file_path, dest_file_path)

origin = Path.cwd().parent / "data" / "files" / "pbmc_cancer"
destination = Path.cwd().parent / "data" / "full-trimmed" / "pbmc_cancer"

df = pd.read_csv("tcrseq_pbmcfnames.csv")
dups = df["LTX_ID"].value_counts().index[(df["LTX_ID"].value_counts() > 2).values]

to_discard = []

for i in dups:
    repeats = df.loc[df["LTX_ID"] == i]["filename"].tolist()
    to_discard += [i for i in repeats if "merge" not in i]

make_directory_where_necessary(destination)
copy_files(origin, destination)

for i in to_discard:
    fname = i.replace(".gz", "")
    try:
        os.remove(destination / fname)
    except FileNotFoundError as e:
        print (f"{destination / fname} not found")
