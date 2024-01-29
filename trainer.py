import torch
from sparsemax import Sparsemax
import numpy as np
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
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
        "batch-size": 512,
        "epoch": 50,
        "lr": [0.01, 0.001, 0.0001],
        "change-lr-at": [10, 25, 40],
        "train-split": 0.8
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
        if epoch == change_epoch:
            return new_lr[change_epoch.index(epoch)] / optimizer.param_groups[0]['lr']
    return 1


class classifier(torch.nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.last_scores = None
        self.last_weights = None

        self.scoring_linear1 = torch.nn.Linear(768, 512)
        self.scoring_linear2 = torch.nn.Linear(512, 128)
        self.scoring_linear3 = torch.nn.Linear(128, 1)
        
        self.relu = torch.nn.LeakyReLU(inplace = False)
        self.sparsemax = Sparsemax(dim = 0)

        self.classifying_linear1 = torch.nn.Linear(768, 512)
        self.classifying_linear2 = torch.nn.Linear(512, 128)
        self.classifying_linear3 = torch.nn.Linear(128, 1)
        
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        self.last_scores = self.scoring_linear3(
            self.relu(self.scoring_linear2(
            self.relu(self.scoring_linear1(x))
            ))
        )

        # There appears to be a bug with the Sparsemax function.
        # Sparsemax only handles 2-dim tensors, therefore there is a code
        # that does:
        # input = input.transpose(0, self.dim)
        # which means if I have dim = 0, no transpose will happen, therefore I
        # need to transpose the scores first and transpose it back.
        self.last_weights = self.sparsemax(self.last_scores.T).T
        agg_out = torch.sum(self.last_weights * x, dim = 0, keepdim = True)

        result = self.classifying_linear3(
            self.relu(self.classifying_linear2(
            self.relu(self.classifying_linear1(agg_out))
            ))
        )
        return self.sig(result)


if __name__ == "__main__":
    try:
        parser = parse_command_line_arguments()
        log = Logger(
            parser.log_file if parser.log_file is not None else "training-logs.log"
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
            torch.save(bertmodel.state_dict(), "bert.pth")
            bertmodel_gpu_usage = torch.cuda.memory_allocated() - start_mem
            log.print(
                f"Memory Reserved for BERT Model: {bertmodel_gpu_usage} bytes"
            )
            log.print("Pushing Classifier Model to cuda")
            start_mem = torch.cuda.memory_allocated()
            classifier_model = classifier_model.to("cuda")
            classifier_gpu_usage = torch.cuda.memory_allocated() - start_mem
            log.print(
                f"Memory Reserved for Classifier Model: {classifier_gpu_usage} bytes"
            )
            
        log.print("Setting Up Optimizer and Loss Function")
        if isinstance(custom_configs["lr"], list) and isinstance(custom_configs["change-lr-at"], list):
            assert len(custom_configs["lr"]) == len(custom_configs["change-lr-at"])
            optimizer = torch.optim.Adam(
                list(bertmodel.parameters()) + list(classifier_model.parameters()),
                lr = custom_configs["lr"][0]
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda = lambda x: lr_lambda(
                    x, custom_configs["change-lr-at"], custom_configs["lr"], optimizer
                )
            )
        elif isinstance(custom_configs["lr"], list):
            log.print(
                "LR and Change LR Epoch are not both lists, the first LR will be used and no scheduling will be done", 
                "WARN"
            )
            optimizer = torch.optim.Adam(
                list(bertmodel.parameters()) + list(classifier_model.parameters()),
                lr = custom_configs["lr"][0]
            )
        
        else:
            log.print(
                "LR and Change LR Epoch are not both lists, Change LR Epoch will be ignored",
                "WARN"
            )
            optimizer = torch.optim.Adam(
                list(bertmodel.parameters()) + list(classifier_model.parameters()),
                lr = custom_configs["lr"]
            )

        
        
        # This really should be BCE, but BCELoss gives nulls after first backprop instance.
        criterion = torch.nn.BCELoss()
        log.print("Optimizer and Loss Set")

        log.print("Loading Patient Data")
        patient_loader = PatientTCRloader(
            split=custom_configs["train-split"], 
            **make_tcrargs(custom_configs),
            shuffle = True
        )
        log.print("Data Loaded")
        log.print("Commencing Training")
        
        trainloss = []
        trainacc  = []

        for e in range(custom_configs["epoch"]):
            batchloss = []
            batchacc  = []
            for i in range(len(patient_loader)):
                filepath, pattcr = patient_loader[i]
                log.print(
                    f"Processing {Path(filepath).stem} ; File {i} / {len(patient_loader)}.  True Label {pattcr[0]}"
                )
                pattcr_loader = torch.utils.data.DataLoader(
                    TCRloader(*pattcr).data,
                    batch_size=custom_configs["batch-size"],
                    shuffle=True,
                )
                
                all_embeddings = []
                for batchno, tcr in enumerate(pattcr_loader):
                    log.print(f"Batch No.: {batchno} / {len(pattcr_loader)}")
                    inputs = tokenizer(tcr, return_tensors="pt", padding=True)

                    if torch.cuda.is_available():
                        log.print(f"Pushing Tokens to cuda")
                        start_mem = torch.cuda.memory_allocated()
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}
                        log.print(
                            f"Memory Reserved for Tokens: {(torch.cuda.memory_allocated() - start_mem)} bytes"
                        )
                        log.print(
                            f"Current CUDA Memory Utilisation (Total): {(torch.cuda.memory_allocated())} bytes"
                        )
                        log.print(
                            f"Current CUDA Memory Utilisation (Excluding BERT Model): {(torch.cuda.memory_allocated()) - bertmodel_gpu_usage} bytes"
                        )
                        log.print(
                            f"Current CUDA Memory Utilisation (Excluding Both Models): {(torch.cuda.memory_allocated()) - classifier_gpu_usage - bertmodel_gpu_usage} bytes"
                        )

                    embeddings = bertmodel(**inputs).last_hidden_state
                    log.print(f"Embeddings Generated")

                    if custom_configs["needed-embedding"] == "last":
                        embeddings = embeddings[:, -1, :]
                    elif custom_configs["needed-embedding"] == "mean":
                        embeddings = torch.mean(embeddings, dim=1)
                    else:
                        raise ValueError(
                            "Unrecognised Needed Configuration Setting.  Supported settings are 'last' and 'mean'"
                        )

                    all_embeddings = all_embeddings + embeddings.tolist()
                    log.print(f"Required Embedding Vectors Extracted")

                    del embeddings, inputs
                    torch.cuda.empty_cache()
                
                log.print(f"All Needed Embeddings Extracted")
                all_embeddings = torch.from_numpy(np.array(all_embeddings)).to(torch.float32)
                all_embeddings = all_embeddings.cuda() if torch.cuda.is_available() else all_embeddings
                prediction     = classifier_model(all_embeddings)
                truelabel      = torch.full_like(prediction, pattcr[0], dtype = torch.float32)
                truelabel      = truelabel.cuda() if torch.cuda.is_available() else truelabel
                loss           = criterion(prediction, truelabel) / patient_loader.ratio(positive = bool(pattcr[0]))
                batchloss.append(loss.data.tolist())
                batchacc.append(int(pattcr[0] == int(prediction >= 0.5)))

                log.print(f"File {i} / {len(patient_loader)}: Predicted Value: {prediction.data.tolist()} ; True Value: {pattcr[0]}")
                log.print(f"File {i} / {len(patient_loader)}: Loss: {loss.data.tolist()}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                del all_embeddings, prediction, truelabel, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            trainloss.append(sum(batchloss) / len(batchloss))
            trainacc.append(sum(batchacc)  / len(batchacc))
            log.print(f"End of Epoch {e}")
            log.print(f"Average Loss: {trainloss[-1]}; Average Accuracy: {trainacc[-1]}")

        pd.DataLoader(trainloss).to_csv("trainloss.csv", index = False, header = False)
        pd.DataLoader(trainacc).to_csv("trainloss.csv", index = False, header = False)
        torch.save(bertmodel.state_dict(), "bert-model-trained.pth")
        torch.save(classifier.state_dict(), "classifier-trained.pth")

    except Exception as e:
        log.print("Saving Model Instance")
        torch.save(bertmodel.state_dict(), "bert-model-mid-trained.pth")
        torch.save(classifier.state_dict(), "classifier-mid-trained.pth")

        log.print(f"Error Encountered: Logging Information", "ERROR")
        if torch.cuda.is_available():
            log.print(f"Torch Memory Taken: {torch.cuda.memory_allocated()}")
        log.print(f"{type(e).__name__}: {str(e)}", "ERROR")
    
    finally:
        try:
            log.close()
        except NameError:
            pass

