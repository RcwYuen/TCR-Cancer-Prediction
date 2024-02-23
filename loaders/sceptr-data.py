import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm


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
        patientid = file.name.split("_")[2]
        chain = file.name.split("_")[-1].replace(".tsv", "")

    else:
        rw = tcrseqfnames.loc[tcrseqfnames["filename"] == file.name + ".gz"]
        patientid = rw["LTX_ID"].values[0]
        chain = rw["chain"].values[0]
        
    if patientid not in list(pairs.keys()):
        pairs[patientid] = [(file, chain)]
    else:
        pairs[patientid].append((file, chain))

required_rows = ["TRAV", 'TRBV', 'TRAJ', 'TRBJ', 'CDR3A', 'CDR3B']

for patientid, pair in tqdm(pairs.items()):
    exportdf = pd.DataFrame()
    for fname, chain in pair:
        if fname.suffix == ".tsv":
            df = pd.read_csv(fname, delimiter="\t")
        else:
            df = pd.read_csv(fname, header=None, usecols=[0])
            df.columns = ["junction_aa"]
            df["v_call"] = None
            df["j_call"] = None

        df = df.dropna(axis=0, how="all")
        exportdf["TRAV" if chain.lower() == "alpha" else "TRBV"] = df["v_call"]
        exportdf["TRAJ" if chain.lower() == "alpha" else "TRBJ"] = df["j_call"]
        exportdf["CDR3A" if chain.lower() == "alpha" else "CDR3B"] = df["junction_aa"]
    
    for r in required_rows:
        if r not in exportdf.columns:
            exportdf[r] = None
    
    exportpath = Path(str(fname.parent).replace(str(org_path), str(des_path)))
    make_directory_where_necessary(exportpath)
    exportdf.to_csv(exportpath / (str(patientid) + ".tsv"), index = False, sep = "\t")
    