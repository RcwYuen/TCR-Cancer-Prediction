import pandas as pd
import shutil, os
from tqdm import tqdm
import gzip, glob
import argparse
import sys
import threading
import json
import re
import warnings
import control_scraper
from pathlib import Path

ARCHIVE_PATH = "data/archive/"


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="Scraper for Chain Lab RDS; Targeting Tx TCRs"
    )
    parser.add_argument("-config_path", help="Path to Filenames CSV")
    parser.add_argument(
        "-U", action="store_true", help="Whether to update the current FILES"
    )
    parser.add_argument(
        "--keep-gz", action="store_true", help="Whether to keep gz compresed FILES"
    )
    parser.add_argument("--silent", action="store_true", help="Report Progress")
    return parser.parse_args()


def relocate_df(files_df, suffix):
    try:
        if UPDATE and os.path.isdir(ARCHIVE_PATH + suffix + "/"):
            shutil.rmtree(ARCHIVE_PATH + suffix)

        if UPDATE or not os.path.isdir(
            ARCHIVE_PATH.replace("archive", "files") + suffix
        ):
            # if update is needed or there are no decompressed files
            os.makedirs(ARCHIVE_PATH + suffix + "/")
            for ltxid, filename in tqdm(
                files_df[["LTX_ID", "filename"]].values.tolist(),
                desc=f"Fetching {suffix} data",
                disable=SILENT,
            ):
                addr = ORG + ltxid + "/" + filename
                shutil.copy2(addr, ARCHIVE_PATH + suffix + "/")
    except FileExistsError:
        pass


def relocate_ls(ls, suffix):
    try:
        if UPDATE and os.path.isdir(ARCHIVE_PATH + suffix + "/"):
            shutil.rmtree(ARCHIVE_PATH + suffix)

        if UPDATE or not os.path.isdir(
            ARCHIVE_PATH.replace("archive", "files") + suffix
        ):
            # if update is needed or there are no decompressed files
            os.makedirs(ARCHIVE_PATH + suffix + "/")
            for filepath in tqdm(
                list(set(ls)), desc=f"Fetching {suffix} data", disable=SILENT
            ):
                shutil.copy2(filepath, ARCHIVE_PATH + suffix + "/")
    except FileExistsError:
        pass


def decompress(extract_from, extract_to):
    try:
        if UPDATE and os.path.isdir(extract_to):
            shutil.rmtree(extract_to)

        os.makedirs(extract_to)
        for fname in tqdm(
            list(glob.glob(extract_from + "*.gz")), desc="Decompressing", disable=SILENT
        ):
            fname = fname.replace("\\", "/")
            with gzip.open(fname, "rt") as f:
                file_content = f.read()
            f = open(extract_to + fname.split("/")[-1].replace(".gz", ""), "w")
            f.write(file_content)
            f.close()

    except FileExistsError:
        pass


def pingrds(drive, event):
    glob.glob(drive)
    event.set()


def vpn_connected(drive):
    event = threading.Event()
    thread = threading.Thread(target=pingrds, args=(drive, event))
    thread.start()
    event.wait(timeout=10)
    connected = not thread.is_alive()
    thread.join()
    return connected


def clean_cache(do, path):
    if do:
        print("Removing Cache")

        try:
            shutil.rmtree(path)
            print("Completed")
        except FileNotFoundError:
            print(f"Unable to find Path {path}, cache clean cancelled")


global FILES, UPDATE, ORG, SILENT

if __name__ == "__main__":
    args = parse_command_line_arguments()

    SILENT = args.silent
    if SILENT:
        sys.stdout = open(os.devnull, "w")

    config_fname = "config.json" if args.config_path is None else args.config_path
    with open(config_fname, "r") as configjson:
        config = json.load(configjson)
        configjson.close()

    mountpoint = config["rds_mountpoint"]
    ORG = f"{mountpoint}/TRACERx_TCRseq_Data_20221015/DATA/TSV/"
    UPDATE = args.U

    print("Checking RDS Connection...")
    if not vpn_connected(mountpoint):
        sys.stdout = sys.__stdout__
        raise ConnectionError(
            "Internet not Connected or UCL VPN not connected.  Please check your connection and try again"
        )
    print("Completed")

    print("Extracting Cancer Data...")
    keys = [k for k in config.keys() if re.match("cancer_.+", k)]
    for key in keys:
        df = pd.read_csv(config[key])
        cancertype = config[f"cancertype_{key[7::]}"]
        relocate_df(df, f"{cancertype}_cancer")
        decompress(
            f"data/archive/{cancertype}_cancer/", f"data/files/{cancertype}_cancer/"
        )
    print("Completed")
    clean_cache(not (args.keep_gz), "data/archive")

    print("Extracting Control Data...")
    keys = [k for k in config.keys() if re.match("control_.+", k)]
    for key in keys:
        control_config = {
            "rds_mountpoint": config["rds_mountpoint"],
            "sample_regex": config[key],
        }
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            control_scraper.main(config=control_config, outdir=Path.cwd())

        with open("all_filepaths.txt", "a+") as f:
            filepaths = open("file_paths.txt", "r")
            f.writelines(filepaths.read() + "\n")
            filepaths.close()
        os.remove("file_paths.txt")

    with open("all_filepaths.txt", "r") as f:
        relocate_ls([i.replace("\n", "") for i in f.readlines()], "control")
        decompress("data/archive/control/", "data/files/control/")
        f.close()
    os.remove("all_filepaths.txt")

    print("Completed")
    clean_cache(not (args.keep_gz), "data/archive")

    sys.stdout = sys.__stdout__
