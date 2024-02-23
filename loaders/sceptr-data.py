import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
import tidytcells as tt
import warnings

tcrseqfnames = pd.read_csv(Path.cwd() / "rds_file_locations" / "TCRseq_filenames.csv")
org_path = Path.cwd() / "data" / "full-trimmed"
des_path = Path.cwd() / "data" / "sceptr"

pairs = {}

def make_directory_where_necessary(directory):
    if not os.path.exists(directory):
        if make_directory_where_necessary(directory.parent):
            os.mkdir(directory)
    return True

for file in org_path.glob("*/*"):
    if "control" in str(file):
        patientid = tuple(file.name.split("_")[:3])
        if file.suffix == ".tsv":
            chain = file.name.split("_")[-1].replace(".tsv", "")
        else:
            chain = file.name.split("_")[-1].replace(".cdr3", "")

    else:
        rw = tcrseqfnames.loc[tcrseqfnames["filename"] == file.name + ".gz"]
        patientid = rw["LTX_ID"].values[0] + "cancer"
        chain = rw["chain"].values[0]
        
    if patientid not in list(pairs.keys()):
        pairs[patientid] = [(file, chain)]
    else:
        pairs[patientid].append((file, chain))

print (pairs)
required_rows = ["TRAV", 'TRBV', 'TRAJ', 'TRBJ', 'CDR3A', 'CDR3B']

for patientid, pair in tqdm(pairs.items()):
    exportdf = {}
    for fname, chain in pair:
        if fname.suffix == ".tsv":
            df = pd.read_csv(fname, delimiter="\t")
        else:
            df = pd.read_csv(fname, header=None, usecols=[0])
            df.columns = ["junction_aa"]
            df["v_call"] = None
            df["j_call"] = None

        df = df.dropna(axis=0, how="all")
        if df["v_call"] is not None:
            exportdf["TRAV" if chain.lower() == "alpha" else "TRBV"] = [
                tt.tr.standardise(i, enforce_functional = True, suppress_warnings = True) if i is not None else 
                i for i in df["v_call"].values
            ]

        if df["j_call"] is not None:
            exportdf["TRAJ" if chain.lower() == "alpha" else "TRBJ"] = [
                tt.tr.standardise(i, enforce_functional = True, suppress_warnings = True) if i is not None else 
                i for i in df["j_call"].values
            ]

        exportdf["CDR3A" if chain.lower() == "alpha" else "CDR3B"] = df["junction_aa"].values.tolist()
    
    for r in required_rows:
        if r not in exportdf.keys():
            exportdf[r] = []

    max_length = max(len(lst) for lst in exportdf.values())
    for key in exportdf:
        exportdf[key] += [None] * (max_length - len(exportdf[key]))
    
    exportpath = Path(str(fname.parent).replace(str(org_path), str(des_path)))
    make_directory_where_necessary(exportpath)
    exportdf = pd.DataFrame(exportdf)
    exportdf.to_csv(exportpath / ("_".join(patientid) + ".tsv"), index = False, sep = "\t")
    