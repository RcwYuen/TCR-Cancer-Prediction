import pandas as pd
from tqdm import tqdm
from pathlib import Path
import os

def make_directory_where_necessary(directory):
    if not os.path.exists(directory):
        if make_directory_where_necessary(directory.parent):
            os.mkdir(directory)
    return True

folders = ["pbmc_cancer", "lung_cancer"]# , "control"]

for dir in [Path.cwd() / "data" / "compressed" / folder for folder in folders]:
    make_directory_where_necessary(dir)


files = [Path.cwd() / "data" / "files" / folder for folder in folders]

files = sum([list(i.glob("*")) for i in files], [])
for i in tqdm(files):
    if i.suffix[1::] == "tsv":
        df = pd.read_csv(i, delimiter = "\t")[["cdr1_aa", "cdr2_aa", "junction_aa"]]
    else: # "cdr3"
        df = pd.read_csv(i, header = None, usecols = [0])

    df = df.dropna(axis=0, how="all")
    df.to_csv(str(i).replace("data\\files", "data\\compressed"), sep = "\t")