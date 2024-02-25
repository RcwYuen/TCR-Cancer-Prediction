import pandas as pd
import numpy as np
from pathlib import Path

def _get_df(fname):
    df = pd.read_csv(fname, delimiter = "\t").set_index("amino.acid").sort_index()
    print (df)
    df = (df.max() - df) / (df.max() - df.min())
    return df


def quantify_repertoire(repertoire, method = "atchley", replace_unrecognised = True):
    if method.lower() == "atchley":
        df = atchley.copy()
    elif method.lower() == "kidera":
        df = kidera.copy()
    elif method.lower() == "aa_prop":
        df = aa_prop.copy()
    
    rep = []
    for tcr in repertoire:
        if tcr is not None:
            tcr = ''.join(filter(lambda char: char in df.index.tolist(), tcr)) if replace_unrecognised else tcr
            if tcr != "":
                rep.append(np.mean(df.loc[list(tcr)].values, axis = 0))
    
    rep = np.array(rep)
    rep = rep[~np.isnan(rep).any(axis = 1)]
    return rep

global atchley, kidera, aa_prop
dir = Path(__file__).resolve().parent
atchley = _get_df(dir / "atchley.txt")
kidera = _get_df(dir / "kidera.txt")
aa_prop = _get_df(dir / "aa_properties.txt")

print (aa_prop)