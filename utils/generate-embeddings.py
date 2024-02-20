import torch, sys
from logger import Logger
from dataloader import PatientTCRloader, TCRloader
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

global log


def parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generator for Embeddings for TCR Data"
    )
    parser.add_argument("-c", "--config-file", help="Location for Configuration File")
    parser.add_argument(
        "--make", action="store_true", help="Make Configuration JSON Template"
    )
    parser.add_argument(
        "--log-file", help="Logging File Name"
    )
    return parser.parse_args()


def default_configs(write_to=False):
    configs = {
        "input-path": "data/files",
        "output-path": "data/embeddings",
        "model-path": "model",
        "maa-model": True,
        "negative-dir": ["control"],
        "positive-dir": ["lung_cancer", "pbmc_cancer"],
        "batch-size": 64,
        "cdr1": False,
        "cdr2": False,
        "needed-embedding": "mean",
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


def make_tcrargs(config):
    return {
        "path": config["input-path"],
        "positives": config["positive-dir"],
        "negatives": config["negative-dir"],
        "cdr1": config["cdr1"],
        "cdr2": config["cdr2"],
    }


def make_output_path(input_path, output_relative_path):
    fname = Path(str(Path(input_path).stem) + ".parquet")
    return Path.cwd() / output_relative_path / input_path.parent.name / fname


def make_directory_where_necessary(directory):
    if not os.path.exists(directory):
        if make_directory_where_necessary(directory.parent):
            os.mkdir(directory)
    return True


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
            model = AutoModelForMaskedLM.from_pretrained(
                custom_configs["model-path"] + "/mlm-only/model"
            ).bert.eval()

        else:
            log.print("Loading Tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(
                custom_configs["model-path"] + "ordinary/tokenizer"
            )
            log.print("Loading Sequence Classification Model")
            model = AutoModelForSequenceClassification.from_pretrained(
                custom_configs["model-path"] + "ordinary/model"
            ).bert.eval()

        if torch.cuda.is_available():
            log.print("Pushing Model to cuda")
            start_mem = torch.cuda.memory_allocated()
            model = model.to("cuda")
            log.print(
                f"Memory Reserved for Model: {(torch.cuda.memory_allocated() - start_mem)} bytes"
            )

        log.print("Loading Patient Data")
        patient_loader = PatientTCRloader(split=1, **make_tcrargs(custom_configs))
        log.print("Data Loaded")
        log.print("Generating Embeddings")

        with torch.no_grad():
            for i in range(len(patient_loader)):
                filepath, pattcr = patient_loader[i]
                log.print(
                    f"Processing {Path(filepath).stem} ; File {i} / {len(patient_loader)}"
                )
                pattcr_loader = torch.utils.data.DataLoader(
                    TCRloader(*pattcr).data,
                    batch_size=custom_configs["batch-size"],
                    shuffle=True,
                )
                all_embeddings = []
                for batchno, tcr in enumerate(pattcr_loader):
                    torch.cuda.empty_cache()
                    log.print(f"Batch No.: {batchno} / {len(pattcr_loader)}")
                    inputs = tokenizer(tcr, return_tensors="pt", padding=True)

                    if torch.cuda.is_available():
                        log.print(f"Pushing Tokens to cuda")
                        start_mem = torch.cuda.memory_allocated()
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}
                        log.print(
                            f"Memory Reserved for Tokens: {(torch.cuda.memory_allocated() - start_mem)} bytes"
                        )

                    embeddings = model(**inputs).last_hidden_state
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

                log.print(f"Embedding Generated for File {Path(filepath).stem}")
                outpath = make_output_path(
                    Path(filepath), custom_configs["output-path"]
                )
                make_directory_where_necessary(outpath.parent)
                log.print(f"Exporting to {outpath}")
                pd.DataFrame(all_embeddings).mean(axis = 0).to_frame().to_parquet(outpath, index=False)
                log.print("All TCRs Converted, freeing cuda memory")
                torch.cuda.empty_cache()
                log.print(f"cuda memory occupied: {torch.cuda.memory_allocated()} b")

        log.print("Done")

    except Exception as e:
        log.print(f"Error Encountered: Logging Information", "ERROR")
        if torch.cuda.is_available():
            log.print(f"Torch Memory Taken: {torch.cuda.memory_allocated()}")
        log.print(f"{type(e).__name__}: {str(e)}", "ERROR")

    finally:
        log.close()
