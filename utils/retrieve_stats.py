from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import numpy as np

def getstats(foldername, rolling_window = 20, endofepoch = False):
    dirs = Path.cwd() / foldername if not isinstance(foldername, Path) else foldername
    epochs = list(dirs.glob("Epoch */*.csv"))
    stats = {"train-acc.csv": [], "train-loss.csv": [], "test-acc.csv": [], "test-loss.csv": []}
    epochwisestats = {"train-acc.csv": [], "train-loss.csv": [], "test-acc.csv": [], "test-loss.csv": []}
    aucstats = {"train-preds.csv": [], "test-preds.csv": []}
    epochwiseauc = {"train-preds.csv": {}, "test-preds.csv": {}}

    maxepochs = -1
    for i in epochs:
        info = str(i).replace(str(dirs) + "\\", "").split("\\")
        info = [int(info[0].replace("Epoch ", "")), info[1]]
        df = pd.read_csv(i, header = None if "pred" not in info[1] else 0) 
        if "loss" in info[1]:
            epochwisestats[info[1]].append(
                (info[0], df.mean().values.tolist()[0]) if not endofepoch else
                  (info[0], df.values[-rolling_window:].mean().tolist())
            )
            df = df.rolling(window = rolling_window).mean().dropna()
            stats[info[1]].append((info[0], df.T.values[0].tolist()))

        elif "acc" in info[1]:
            epochwisestats[info[1]].append((info[0], df.mean().values.tolist()[0]))
            df = df.rolling(window = rolling_window).mean().dropna()
            stats[info[1]].append((info[0], df.T.values[0].tolist()))

        elif "pred" in info[1]:
            epochwiseauc[info[1]][info[0]] = roc_curve(df["actual"], df["preds"])
            aucstats[info[1]].append((info[0], roc_auc_score(df["actual"], df["preds"])))

        maxepochs = max(maxepochs, info[0])

    for key in stats.keys():
        stats[key].sort(key = lambda x: x[0])
        stats[key] = sum([i[1] for i in stats[key]], [])

    for key in aucstats.keys():
        aucstats[key].sort(key = lambda x: x[0])
        aucstats[key] = [i[1] for i in aucstats[key]]

    for key in epochwisestats.keys():
        epochwisestats[key].sort(key = lambda x: x[0])
        epochwisestats[key] = [i[1] for i in epochwisestats[key]]
    
    return {
        "stats": stats,
        "epochwisestats": epochwisestats,
        "aucstats": aucstats,
        "epochwiseauc": epochwiseauc
    }

def find_bestepoch(stats, focus_last = True, avoid_premature = 0):
    auc = stats["aucstats"]["test-preds.csv"][avoid_premature:]
    auc = auc[::-1] if focus_last else auc
    return len(auc) - np.argmax(auc) - 1 + avoid_premature if focus_last else np.argmax(auc) + avoid_premature