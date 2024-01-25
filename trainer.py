import torch
from sparsemax import Sparsemax
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from utils.logger import Logger
from utils.dataloader import PatientTCRloader, TCRloader
import argparse
import json
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
)
import pandas as pd
from pathlib import Path
import os
import sys

global log



def parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model Trainer for TCR Data")
    parser.add_argument("-c", "--config-file", help="Location for Configuration File")
    parser.add_argument(
        "--make", action="store_true", help="Make Configuration JSON Template"
    )
    parser.add_argument("--log-file", help="Logging File Name")
    return parser.parse_args()


def default_configs(write_to=False):
    configs = {
        "input-path": "data/files",
        "output-path": "model/trained",
        "model-path": "model",
        "maa-model": True,
        "negative-dir": ["control"],
        "positive-dir": ["lung_cancer", "pbmc_cancer"],
        "cdr1": False,
        "cdr2": False,
        "needed-embedding": "mean",
        "batch-size": 1024,
        "epoch": 50,
        "lr": [0.01, 0.001, 0.0001],
        "change-lr-at": [10, 25, 40],
    }

    if write_to:
        with open("config.json", "w") as outs:
            outs.write(json.dumps(configs, indent=4))

    return configs


def load_configs(custom_configs):
    configs = default_configs()
    for key, val in custom_configs.items():
        log.print(f"Config: {key}: {val}", "INFO")
        if key in configs:
            configs[key] = val
        else:
            log.print(
                f"Unrecognised Configuration Found.  Please regenerate the configuration file with 'python {arg} --make'"
            )
            raise ValueError(f"Unrecongised Configuration Found: {key}")
    return configs


def make_directory_where_necessary(directory):
    if not os.path.exists(directory):
        if make_directory_where_necessary(directory.parent):
            os.mkdir(directory)
    return True


def make_tcrargs(config):
    return {
        "path": config["input-path"],
        "positives": config["positive-dir"],
        "negatives": config["negative-dir"],
        "cdr1": config["cdr1"],
        "cdr2": config["cdr2"],
    }


def lr_lambda(epoch, change_epochs, new_lrs, optimizer):
    for change_epoch, new_lr in zip(change_epochs, new_lrs):
        if epoch in change_epoch:
            return new_lr[change_epoch.index(epoch)] / optimizer.param_groups[0]['lr']
    return 1


if __name__ == "__main__":
    try:
        parser = parse_command_line_arguments()
        log = Logger(
            parser.log_file if parser.log_file is not None else "embedding-logs.log"
        )
        log.print("Instanciating")
        arg = " ".join(sys.argv)
        log.print("Arguments: python " + " ".join(sys.argv), "INFO", silent=True)
        if parser.make:
            log.print("Creating Configuration Template")
            default_configs(write_to=True)
            quit()

        config_file = (
            parser.config_file if parser.config_file is not None else "config.json"
        )
        custom_configs = load_configs(json.load(open(config_file, "r")))
        log.print("Tokenizer Loaded")
        if custom_configs["maa-model"]:
            log.print("Loading Tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(
                custom_configs["model-path"] + "/mlm-only/tokenizer"
            )
            log.print("Loading Masked Amino Acid Model")
            bertmodel = AutoModelForMaskedLM.from_pretrained(
                custom_configs["model-path"] + "/mlm-only/model"
            ).bert

        else:
            log.print("Loading Tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(
                custom_configs["model-path"] + "ordinary/tokenizer"
            )
            log.print("Loading Sequence Classification Model")
            bertmodel = AutoModelForSequenceClassification.from_pretrained(
                custom_configs["model-path"] + "ordinary/model"
            ).bert

        classifier_model = classifier()
        if torch.cuda.is_available():
            log.print("Pushing BERT Model to cuda")
            start_mem = torch.cuda.memory_allocated()
            bertmodel = bertmodel.to("cuda")
            log.print(
                f"Memory Reserved for BERT Model: {(torch.cuda.memory_allocated() - start_mem)} bytes"
            )
            log.print("Pushing Classifier Model to cuda")
            start_mem = torch.cuda.memory_allocated()
            classifier_model = classifier_model.to("cuda")
            log.print(
                f"Memory Reserved for Classifier Model: {(torch.cuda.memory_allocated() - start_mem)} bytes"
            )
            
        log.print("Setting Up Optimizer and Loss Function")
        assert isinstance(custom_configs["lr"], type(custom_configs["change-lr-at"]))
        assert len(custom_configs["lr"]) == len(custom_configs["change-lr-at"])
        custom_configs["lr"] = [custom_configs["lr"]] \
            if isinstance(custom_configs["lr"], float) else custom_configs["lr"]
        custom_configs["change-lr-at"] = [custom_configs["change-lr-at"]] \
            if isinstance(custom_configs["change-lr-at"], float) else custom_configs["change-lr-at"]
        
        optimizer = torch.optim.Adam(
            list(bertmodel.parameters) + len(classifier_model),
            lr = custom_configs["lr"][0]
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda = lambda x: lr_lambda(
                x, custom_configs["change-lr-at"], custom_configs["lr"], optimizer
            )
        )
        criterion = torch.nn.BCELoss()
        log.print("Optimizer and Loss Set")

        log.print("Loading Patient Data")
        patient_loader = PatientTCRloader(split=1, **make_tcrargs(custom_configs))
        log.print("Data Loaded")
        log.print("Commencing Training")
        
        for e in range(custom_configs["epoch"]):
            pass

    except Exception as e:
        log.print(f"Error Encountered: Logging Information", "ERROR")
        if torch.cuda.is_available():
            log.print(f"Torch Memory Taken: {torch.cuda.memory_allocated()}")
        log.print(f"{type(e).__name__}: {str(e)}", "ERROR")

    finally:
        log.close()

