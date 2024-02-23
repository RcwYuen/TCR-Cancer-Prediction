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
from model import oneinner as classifier
import time
import datetime
import gc

global log, total_patients

def parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model Trainer for TCR Data")
    parser.add_argument("-c", "--config-file", help="Location for Configuration File")
    parser.add_argument(
        "--make", action="store_true", help="Make Configuration JSON Template"
    )
    parser.add_argument("--log-file", help="Logging File Name")
    parser.add_argument(
        "--end", action="store_true", 
        help = "End after making Configuration Template.  Not applied \
            if Config Template is not going to be produce.")
    return parser.parse_args()


def default_configs(write_to=False):
    configs = {
        "input-path": "data/files",
        "output-path": "trained",
        "model-path": "model",
        "maa-model": True,
        "negative-dir": ["control"],
        "positive-dir": ["pbmc_cancer"],
        "cdr1": False,
        "cdr2": False,
        "batch-size": 512,
        "epoch": 50,
        "lr": 0.001,
        "change-lr-at": 50,
        "train-split": 0.8,
        "bag-accummulate-loss": 4,
        "l2-penalty": 0
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
        "split": config["train-split"]
    }


def lr_lambda(epoch, change_epochs, new_lrs, optimizer):
    for change_epoch, new_lr in zip(change_epochs, new_lrs):
        if epoch == change_epoch:
            return new_lr[change_epoch.index(epoch)] / optimizer.param_groups[0]['lr']
    return 1


def load_llm(custom_configs):
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
    return tokenizer, bertmodel


def throw_to_cuda(models, names):
    if torch.cuda.is_available():
        cuda_models = []
        model_gpu_usage = []
        for m, n in zip(models, names):
            cuda_models.append(m)
            log.print(f"Pushing {n} Model to cuda")
            start_mem = torch.cuda.memory_allocated()
            cuda_models[-1] = cuda_models[-1].to("cuda")
            gpu_usage = torch.cuda.memory_allocated() - start_mem
            log.print(
                f"Memory Reserved for {n} Model: {gpu_usage} bytes"
            )
            model_gpu_usage.append(gpu_usage)
        return cuda_models, model_gpu_usage
    return models, [None] * len(models)


def make_optimizer(custom_configs):
    scheduler = None
    trainable_params = list(classifier_model.parameters())
    if isinstance(custom_configs["lr"], list) and isinstance(custom_configs["change-lr-at"], list):
        assert len(custom_configs["lr"]) == len(custom_configs["change-lr-at"])
        optimizer = torch.optim.Adam(
            trainable_params,
            lr = custom_configs["lr"][0],
            weight_decay = custom_configs["l2-penalty"]
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
            trainable_params,
            lr = custom_configs["lr"][0],
            weight_decay = custom_configs["l2-penalty"]
        )
    else:
        log.print(
            "LR and Change LR Epoch are not both lists, Change LR Epoch will be ignored",
            "WARN"
        )
        optimizer = torch.optim.Adam(
            trainable_params,
            lr = custom_configs["lr"],
            weight_decay = custom_configs["l2-penalty"]
        )

    return optimizer, scheduler


def projected_completion_time(start_time, custom_configs, current_status):
    epochno, curpat = current_status
    batches_done = epochno * total_patients + curpat + 1
    batches_need = (custom_configs["epoch"] - epochno + 1) * total_patients - curpat
    return batches_need * (time.time() - start_time) / batches_done


def make_output_path(outdir, epochno = "", fname = ""):
    if epochno != "":
        return Path.cwd() / outdir / f"Epoch {epochno}" / fname
    else:
        return Path.cwd() / outdir


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
            if parser.end:
                quit()

        config_file = (
            parser.config_file if parser.config_file is not None else "config.json"
        )

        if torch.cuda.is_available():
            log.print(f"Torch CUDA Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                log.print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")

        # In case config.json does not exist
        custom_configs = default_configs(write_to=False) 
        custom_configs = load_configs(json.load(open(config_file, "r")))
        tokenizer, bertmodel = load_llm(custom_configs)
        classifier_model = classifier()
        models, gpu_usages = throw_to_cuda(
            [classifier_model, bertmodel],
            ["Classifier", "BERT"]
        )
        classifier_model, bertmodel = models
        classifier_gpu_usage, bertmodel_gpu_usage = gpu_usages

        for param in bertmodel.parameters():
            param.requires_grad = False

        log.print("Setting Up Optimizer and Loss Function")
        optimizer, scheduler = make_optimizer(custom_configs)
        optimizer.zero_grad()
        criterion = torch.nn.BCELoss()
        log.print("Optimizer and Loss Set")

        log.print("Loading Patient Data")
        patient_loader = PatientTCRloader(
            **make_tcrargs(custom_configs),
            shuffle = True
        )
        total_patients = len(patient_loader)
        log.print("Data Loaded")
        log.print("Commencing Training")
        
        trainloss = []
        trainacc  = []
        testloss  = []
        testacc   = []
        start_time = time.time()
        for e in range(custom_configs["epoch"]):
            trainbatchloss = []
            trainbatchacc  = []
            testbatchloss  = []
            testbatchacc   = []
            trainactualpreds = {"preds": [], "actual": [], "tcr-count": []}
            testactualpreds  = {"preds": [], "actual": [], "tcr-count": []}
            outpath = make_output_path(custom_configs["output-path"], e)
            make_directory_where_necessary(outpath)
            accummulatedloss = []
            patient_loader.set_mode(train = True)
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
                    embeddings = torch.mean(embeddings, dim=1)
                    log.print(f"Embeddings Generated")

                    all_embeddings = all_embeddings + embeddings.tolist()
                    log.print(f"Required Embedding Vectors Extracted")
                    del inputs
                    torch.cuda.empty_cache()
                    gc.collect()

                trainactualpreds["tcr-count"].append(len(all_embeddings))

                log.print(f"All Needed Embeddings Extracted")
                all_embeddings = torch.from_numpy(np.array(all_embeddings)).to(torch.float32)
                all_embeddings = all_embeddings.cuda() if torch.cuda.is_available() else all_embeddings
                prediction     = classifier_model(all_embeddings)
                truelabel      = torch.full_like(prediction, pattcr[0], dtype = torch.float32)
                truelabel      = truelabel.cuda() if torch.cuda.is_available() else truelabel
                loss           = criterion(prediction, truelabel) / patient_loader.ratio(positive = bool(pattcr[0]))
                lossval        = criterion(prediction, truelabel).data.tolist()

                accummulatedloss.append(lossval)
                trainbatchloss.append(lossval)
                trainactualpreds["preds"].append(prediction.data.tolist()[0][0])
                trainactualpreds["actual"].append(pattcr[0])
                trainbatchacc.append(int(pattcr[0] == int(prediction >= 0.5)))

                loss.backward()

                log.print(f"File {i} / {len(patient_loader)}: Predicted Value: {prediction.data.tolist()[0][0]} ; True Value: {pattcr[0]}")
                log.print(f"File {i} / {len(patient_loader)}: Loss: {lossval}")
                secs_needed = projected_completion_time(start_time, custom_configs, [e, i])
                log.print(f"Projected Time Needed: {secs_needed} seconds")
                log.print(f"Projected Completion Time: {str(datetime.datetime.now() + datetime.timedelta(seconds = secs_needed))}")

                if (i + 1) % custom_configs["bag-accummulate-loss"] == 0 or i == (len(patient_loader) - 1):
                    log.print(f"Updating Network")
                    log.print(f"Accummulated Loss (Unnormalised): {str([round(i, 4) for i in accummulatedloss])}")
                    log.print(f"Accummulated Loss (Average): {sum(accummulatedloss) / len(accummulatedloss)}")
                    optimizer.step()
                    log.print(f"Clearing Gradients")
                    optimizer.zero_grad()
                    accummulatedloss = []

                del all_embeddings, prediction, truelabel, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            
            pd.DataFrame(trainbatchloss).to_csv(outpath / "train-loss.csv", index = False, header = False)
            pd.DataFrame(trainbatchacc).to_csv(outpath / "train-acc.csv", index = False, header = False)
            pd.DataFrame(trainactualpreds).to_csv(outpath / "train-preds.csv", index = False)
            trainloss.append(sum(trainbatchloss) / len(trainbatchloss))
            trainacc.append(sum(trainbatchacc)  / len(trainbatchacc))

            log.print("Validating", "INFO")
            patient_loader.set_mode(train = False)
            with torch.no_grad():
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
                        embeddings = torch.mean(embeddings, dim=1)
                        log.print(f"Embeddings Generated")

                        all_embeddings = all_embeddings + embeddings.tolist()
                        log.print(f"Required Embedding Vectors Extracted")
                        del embeddings, inputs
                        torch.cuda.empty_cache()
                        gc.collect()

                    testactualpreds["tcr-count"].append(len(all_embeddings))
                    log.print(f"All Needed Embeddings Extracted")
                    all_embeddings = torch.from_numpy(np.array(all_embeddings)).to(torch.float32)
                    all_embeddings = all_embeddings.cuda() if torch.cuda.is_available() else all_embeddings
                    prediction     = classifier_model(all_embeddings)
                    truelabel      = torch.full_like(prediction, pattcr[0], dtype = torch.float32)
                    truelabel      = truelabel.cuda() if torch.cuda.is_available() else truelabel
                    loss           = criterion(prediction, truelabel)
                    testbatchloss.append(loss.data.tolist())
                    testbatchacc.append(int(pattcr[0] == int(prediction >= 0.5)))
                    testactualpreds["preds"].append(prediction.data.tolist()[0][0])
                    testactualpreds["actual"].append(pattcr[0])

                    log.print(f"File {i} / {len(patient_loader)}: Predicted Value: {prediction.data.tolist()[0][0]} ; True Value: {pattcr[0]}")
                    log.print(f"File {i} / {len(patient_loader)}: Loss: {loss.data.tolist()}")
                    secs_needed = projected_completion_time(start_time, custom_configs, [e, i])
                    log.print(f"Projected Time Needed: {secs_needed} seconds")
                    log.print(f"Projection Completion Time: {str(datetime.datetime.now() + datetime.timedelta(seconds = secs_needed))}")

                    del all_embeddings, prediction, truelabel, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()

            pd.DataFrame(testbatchloss).to_csv(outpath / "test-loss.csv", index = False, header = False)
            pd.DataFrame(testbatchacc).to_csv(outpath / "test-acc.csv", index = False, header = False)
            pd.DataFrame(testactualpreds).to_csv(outpath / "test-preds.csv", index = False)
            torch.save(classifier_model.state_dict(), outpath / f"classifier-{e}.pth")

            log.print(f"End of Epoch {e}")
            log.print(f"Average Loss: {trainloss[-1]}; Average Accuracy: {trainacc[-1]}")
            testloss.append(sum(testbatchloss) / len(testbatchloss))
            testacc.append(sum(testbatchacc)  / len(testbatchacc))

        # Generate AUC Statistics
        log.print("Generating Predictions for AUC", "INFO")
        outpath = make_output_path(custom_configs["output-path"])
        make_directory_where_necessary(outpath)
        
        with torch.no_grad():
            trainbatchloss = []
            trainbatchacc  = []
            testbatchloss  = []
            testbatchacc   = []
            trainactualpreds = {"preds": [], "actual": [], "tcr-count": []}
            testactualpreds  = {"preds": [], "actual": [], "tcr-count": []}
            patient_loader.set_mode(train = True)
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
                    embeddings = torch.mean(embeddings, dim=1)
                    log.print(f"Embeddings Generated")

                    all_embeddings = all_embeddings + embeddings.tolist()
                    log.print(f"Required Embedding Vectors Extracted")
                    del embeddings, inputs
                    torch.cuda.empty_cache()
                    gc.collect()
                
                log.print(f"All Needed Embeddings Extracted")
                trainactualpreds["tcr-count"].append(len(all_embeddings))
                all_embeddings = torch.from_numpy(np.array(all_embeddings)).to(torch.float32)
                all_embeddings = all_embeddings.cuda() if torch.cuda.is_available() else all_embeddings
                prediction     = classifier_model(all_embeddings)
                truelabel      = torch.full_like(prediction, pattcr[0], dtype = torch.float32)
                truelabel      = truelabel.cuda() if torch.cuda.is_available() else truelabel
                loss           = criterion(prediction, truelabel) / patient_loader.ratio(positive = bool(pattcr[0]))
                lossval        = criterion(prediction, truelabel).data.tolist()

                trainbatchloss.append(lossval)
                trainactualpreds["preds"].append(prediction.data.tolist()[0][0])
                trainactualpreds["actual"].append(pattcr[0])
                trainbatchacc.append(int(pattcr[0] == int(prediction >= 0.5)))
                log.print(f"File {i} / {len(patient_loader)}: Predicted Value: {prediction.data.tolist()[0][0]} ; True Value: {pattcr[0]}")
                log.print(f"File {i} / {len(patient_loader)}: Loss: {lossval}")

                del all_embeddings, prediction, truelabel, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            
            pd.DataFrame(trainbatchloss).to_csv(outpath / "train-set-loss.csv", index = False, header = False)
            pd.DataFrame(trainbatchacc).to_csv(outpath / "train-set-acc.csv", index = False, header = False)
            pd.DataFrame(trainactualpreds).to_csv(outpath / "train-set-preds.csv", index = False)

            log.print("Validation Set", "INFO")
            patient_loader.set_mode(train = False)
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
                    embeddings = torch.mean(embeddings, dim=1)
                    log.print(f"Embeddings Generated")

                    all_embeddings = all_embeddings + embeddings.tolist()
                    log.print(f"Required Embedding Vectors Extracted")
                    del embeddings, inputs
                    torch.cuda.empty_cache()
                    gc.collect()

                log.print(f"All Needed Embeddings Extracted")
                testactualpreds["tcr-count"].append(len(all_embeddings))
                all_embeddings = torch.from_numpy(np.array(all_embeddings)).to(torch.float32)
                all_embeddings = all_embeddings.cuda() if torch.cuda.is_available() else all_embeddings
                prediction     = classifier_model(all_embeddings)
                truelabel      = torch.full_like(prediction, pattcr[0], dtype = torch.float32)
                truelabel      = truelabel.cuda() if torch.cuda.is_available() else truelabel
                loss           = criterion(prediction, truelabel)
                testbatchloss.append(loss.data.tolist())
                testbatchacc.append(int(pattcr[0] == int(prediction >= 0.5)))
                testactualpreds["preds"].append(prediction.data.tolist()[0][0])
                testactualpreds["actual"].append(pattcr[0])

                log.print(f"File {i} / {len(patient_loader)}: Predicted Value: {prediction.data.tolist()[0][0]} ; True Value: {pattcr[0]}")
                log.print(f"File {i} / {len(patient_loader)}: Loss: {loss.data.tolist()}")
                secs_needed = projected_completion_time(start_time, custom_configs, [e, i])
                log.print(f"Projected Time Needed: {secs_needed} seconds")
                log.print(f"Projection Completion Time: {str(datetime.datetime.now() + datetime.timedelta(seconds = secs_needed))}")

                del all_embeddings, prediction, truelabel, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

            pd.DataFrame(testbatchloss).to_csv(outpath / "test-set-loss.csv", index = False, header = False)
            pd.DataFrame(testbatchacc).to_csv(outpath / "test-set-acc.csv", index = False, header = False)
            pd.DataFrame(testactualpreds).to_csv(outpath / "test-set-preds.csv", index = False)


    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        log.print(f"Error Encountered: Logging Information", "ERROR")
        if torch.cuda.is_available():
            log.print(f"Torch Memory Taken: {torch.cuda.memory_allocated()}")
        log.print(f"Line {exc_tb.tb_lineno} - {type(e).__name__}: {str(e)}", "ERROR")
    
    except KeyboardInterrupt:
        log.print("Interrupted", "INFO")

    finally:
        try:
            outpath = make_output_path(custom_configs["output-path"])
            make_directory_where_necessary(outpath)
            log.close()
            pd.DataFrame(trainloss).to_csv(outpath / "trainloss.csv", index = False, header = False)
            pd.DataFrame(trainacc).to_csv(outpath / "trainacc.csv", index = False, header = False)
            torch.save(classifier_model.state_dict(), outpath / "classifier-trained.pth")
        except NameError:
            pass

