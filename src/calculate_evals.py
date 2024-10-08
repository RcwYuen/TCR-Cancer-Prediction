from pathlib import Path
from sceptr import sceptr
from model import sceptr_unidirectional, load_trained
from tqdm import tqdm
import warnings
import pandas as pd
import torch


warnings.simplefilter("ignore")

bestepoch = 49

path = Path.cwd() / "results" / "sceptr" / "eval" / f"trained-sceptr-caneval-4"
evalpath = Path.cwd() / "data" / "sceptr-eval"
model = load_trained(
    path / f"Epoch {bestepoch}" / f"classifier-{bestepoch}.pth",
    sceptr_unidirectional
)
preds = {
    "preds": [],
    "actual": [],
}
for file in tqdm(list(evalpath.glob("*/*.tsv"))):
    df = pd.read_csv(file, sep = "\t")
    embedding = sceptr.calc_vector_representations(df)
    embedding = torch.from_numpy(embedding).cuda() if torch.cuda.is_available() else torch.from_numpy(embedding)
    preds["preds"].append(model(embedding).item())
    preds["actual"].append(1 if "cancer" in str(file) else 0)
pd.DataFrame(preds).to_csv(path / f"eval-set-auc-{bestepoch}.csv", index = False)