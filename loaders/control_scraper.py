"""
Script to scrape Decombinator output from the Chain lab RDS.

** Note **
Work on this script is fully credited to Yuta Nagano's previous
work.  Some slight modification has been made to customise this
program for `load.py`.
"""

import argparse
import json
import pandas as pd
from pandas import DataFrame
from pathlib import Path
import re
from re import Pattern
from warnings import warn


def parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Decombinator output from the Chain lab RDS."
    )
    parser.add_argument(
        "-o", "--output-directory", help="Directory where program will dump its output."
    )
    parser.add_argument("config_path", help="Path to the config json file.")
    return parser.parse_args()


def read_template(rds_root: Path) -> DataFrame:
    """
    Look inside the TCR template directory and load the latest template file.
    """
    template_dir = rds_root / "TcR_Template"

    xlsx_files = [p for p in template_dir.iterdir() if p.suffix == ".xlsx"]
    xlsx_files = sorted(xlsx_files, key=lambda x: x.stat().st_mtime)
    most_recent_template = xlsx_files[-1]

    return pd.read_excel(most_recent_template)


def get_exp_ids(template: DataFrame, sample_regex: Pattern) -> list:
    """
    Filter template file based on sample regex, and fetch all relevant
    experiment IDs.

    NOTE: sample IDs are forced to lowercase before matching.
    NOTE: experiment IDs are forced to lowercase before returning.
    """
    sample_filter = (
        template["Seq_File_ID"]
        .str.lower()
        .str.contains(sample_regex, na=False, regex=True)
    )
    exp_ids = template["Experiment_ID"][sample_filter].unique().tolist()

    def generate_exp_id_variants(exp_id):
        match = re.match("^([a-z]+)(\d+)$", exp_id.lower())
        if not match:
            return [exp_id.lower()]

        exp_name, exp_num = match.group(1), match.group(2)
        exp_num = int(exp_num)
        variants = [
            f"{exp_name}{exp_num:01d}",
            f"{exp_name}{exp_num:02d}",
            f"{exp_name}{exp_num:03d}",
        ]

        return variants

    exp_ids_processed = []
    for exp_id in exp_ids:
        exp_ids_processed.extend(generate_exp_id_variants(exp_id))

    # Return experiment IDs, alphanumerically ordered
    return sorted(exp_ids_processed)


def get_tsv_paths(rds_root: Path, sample_regex: Pattern, exp_ids: list) -> list:
    """
    Fetch paths to all relevant Decombinator output tsv files given the sample
    regex and the relevant experiment IDs.
    """
    processed_dir = rds_root / "TcRSeq_Processed"
    exp_dirs = []
    exp_files = []

    for exp_id in exp_ids:
        dirs = [
            p / "Translated"
            for p in processed_dir.iterdir()
            if exp_id in p.stem.lower() and (p / "Translated").is_dir()
        ]

        if len(dirs) == 0:
            warn(f"No matching directories found for experiment {exp_id}")
            continue

        exp_dirs.extend(dirs)

    for exp_dir in exp_dirs:
        files = [
            p
            for p in exp_dir.iterdir()
            if p.is_file() and sample_regex.search(p.stem.lower())
        ]

        if len(files) == 0:
            warn(f"No matching files found in experiment folder {exp_dir.stem}")
            continue

        exp_files.extend(files)

    return sorted(exp_files)


def save_to_text_file(outdir: Path, file_paths: list) -> None:
    with open(outdir / "file_paths.txt", "w") as f:
        f.writelines("\n".join([str(p) for p in file_paths]))


def main(config: dict, outdir: Path) -> None:
    rds_root = Path(config["rds_mountpoint"] + "\\")
    sample_regex = re.compile(config["sample_regex"])

    print("Loading Template...")
    template = read_template(rds_root)

    print("Fetching Experiment IDs...")
    exp_ids = get_exp_ids(template, sample_regex)

    print("Fetching Decombinator Files of Interest...")
    file_paths = get_tsv_paths(rds_root, sample_regex, exp_ids)

    print("Saving to Text File...")
    save_to_text_file(outdir, file_paths)


if __name__ == "__main__":
    args = parse_command_line_arguments()

    if args.output_directory is None:
        outdir = Path.cwd()
    else:
        outdir = Path(args.output_directory).resolve()

    if not outdir.is_dir():
        raise FileNotFoundError(
            f"The specified output directory does not exist: {outdir}"
        )

    with open(args.config_path, "r") as f:
        config = json.load(f)

    main(config=config, outdir=outdir)
