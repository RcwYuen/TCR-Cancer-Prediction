import re
from pathlib import Path

base_path = Path.cwd() / "data" / "trimmed"
pbmc_cancer_path = base_path / "pbmc_cancer"
control_path = base_path / "control"

# Function to prepend the correct directory and check for file extensions
def __get_full_path(pattern):
    filepath = pbmc_cancer_path / pattern.group(1) if pattern.group(2) == "1" else \
                control_path / pattern.group(1)
    tsv = filepath.with_suffix(".tsv")
    cdr3 = filepath.with_suffix(".cdr3")
    return tsv if tsv.exists() else cdr3


def train_test_datasets(log_file_path):
    file_mention_pattern = re.compile(r'\[.*?\]: Processing (.*?) ; File \d+ / \d+\.  True Label (\d+)')
    validating_pattern = re.compile(r'\[.*?\]: \[INFO\] Validating')
    end_of_epoch_0_pattern = re.compile(r'\[.*?\]: End of Epoch 0')
    training_files = []
    validation_files = []
    found_validation_marker = False

    with open(log_file_path, 'r') as file:
        for line in file:
            if end_of_epoch_0_pattern.search(line):
                break

            if validating_pattern.search(line):
                found_validation_marker = True
                continue

            if not found_validation_marker:
                pattern = file_mention_pattern.search(line)
                if pattern:
                    training_files.append(__get_full_path(pattern))

            else:
                pattern = file_mention_pattern.search(line)
                if pattern:
                    validation_files.append(__get_full_path(pattern))
                        

    unique_training_files = sorted(set(training_files))
    unique_validation_files = sorted(set(validation_files))

    return unique_training_files, unique_validation_files
