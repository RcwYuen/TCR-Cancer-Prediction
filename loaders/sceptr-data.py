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

def cleandf(df):
    # We enforce the V call and J call to be from Alpha or Beta Chains First
    enforce_abv = df["v_call"].str.startswith("TRA") | df["v_call"].str.startswith("TRB") | pd.isna(df["v_call"])
    enforce_abj = df["j_call"].str.startswith("TRA") | df["j_call"].str.startswith("TRB") | pd.isna(df["j_call"])
    enforce_cdr3_notempty = ~pd.isna(df["junction_aa"])
    df = df[enforce_abv & enforce_abj & enforce_cdr3_notempty].copy()
    # We then enforce them to be functional
    df["v_call"] = df["v_call"].apply(lambda i: tt.tr.standardise(i, enforce_functional = True, suppress_warnings = True) \
        if i is not None else i)

    df["j_call"] = df["j_call"].apply(lambda i: tt.tr.standardise(i, enforce_functional = True, suppress_warnings = True) \
        if i is not None else i)
    
    return df

def make_directory_where_necessary(directory):
    if not os.path.exists(directory):
        if make_directory_where_necessary(directory.parent):
            os.mkdir(directory)
    return True


for file in org_path.glob("*/*"):
    if "control" in str(file):
        patientid = "_".join(tuple(file.name.split("_")[:3]))
        if file.suffix == ".tsv":
            chain = file.name.split("_")[-1].replace(".tsv", "")
        else:
            chain = file.name.split("_")[-1].replace(".cdr3", "")

    else:
        rw = tcrseqfnames.loc[tcrseqfnames["filename"] == file.name + ".gz"]
        patientid = rw["LTX_ID"].values[0] + "_positive"
        chain = rw["chain"].values[0]
    
    if patientid not in list(pairs.keys()):
        pairs[patientid] = [(file, chain)]
    else:
        pairs[patientid].append((file, chain))

required_rows = ['TRAV', 'TRBV', 'TRAJ', 'TRBJ', 'CDR3A', 'CDR3B']

for patientid, pair in tqdm(pairs.items()):
    exportdf = {"TRAV": [], "TRAJ": [], "CDR3A": [], "TRBV": [], "TRBJ": [], "CDR3B": []}
    for fname, chain in pair:
        if fname.suffix == ".tsv":
            df = pd.read_csv(fname, delimiter="\t")
        else:
            df = pd.read_csv(fname, header=None, usecols=[0])
            df.columns = ["junction_aa"]
            df["v_call"] = None
            df["j_call"] = None
        
        df = df[["v_call", "j_call", "junction_aa"]]
        df = df.dropna(axis=0, how="all")
        df = cleandf(df)
        if chain.lower() == "alpha":
            exportdf["TRAV"]  += df["v_call"].values.tolist()
            exportdf["TRAJ"]  += df["j_call"].values.tolist()
            exportdf["CDR3A"] += df["junction_aa"].values.tolist()
            exportdf["TRBV"]  += [""] * len(df["v_call"].values.tolist())
            exportdf["TRBJ"]  += [""] * len(df["j_call"].values.tolist())
            exportdf["CDR3B"] += [""] * len(df["junction_aa"].values.tolist())
        else:
            exportdf["TRBV"]  += df["v_call"].values.tolist()
            exportdf["TRBJ"]  += df["j_call"].values.tolist()
            exportdf["CDR3B"] += df["junction_aa"].values.tolist()
            exportdf["TRAV"]  += [""] * len(df["v_call"].values.tolist())
            exportdf["TRAJ"]  += [""] * len(df["j_call"].values.tolist())
            exportdf["CDR3A"] += [""] * len(df["junction_aa"].values.tolist())
    
    for r in required_rows:
        if r not in exportdf.keys():
            exportdf[r] = []
    
    exportpath = Path(str(fname.parent).replace(str(org_path), str(des_path)))
    make_directory_where_necessary(exportpath)
    exportdf = pd.DataFrame(exportdf)
    exportdf.to_csv(exportpath / (f"{patientid}.tsv"), index = False, sep = "\t")