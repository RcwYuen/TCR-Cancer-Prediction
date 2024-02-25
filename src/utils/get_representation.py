import pandas as pd
import numpy as np
from pathlib import Path

def _get_df(fname):
    df = pd.read_csv(fname, delimiter = "\t").set_index("amino.acid").sort_index()
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
            rep.append(np.mean(df.loc[list(tcr)].values, axis = 0))
    return np.array(rep)

global atchley, kidera, aa_prop
dir = Path(__file__).resolve().parent
atchley = _get_df(dir / "atchley.txt")
kidera = _get_df(dir / "aa_properties.txt")
aa_prop = _get_df(dir / "kidera.txt")
print (atchley)