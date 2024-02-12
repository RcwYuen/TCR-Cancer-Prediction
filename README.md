# Multi-Instance Transfer Learning on TCR-BERT for Cancer Prediction

This project aims to apply transfer learning to [TCR-BERT](https://www.biorxiv.org/content/10.1101/2021.11.18.469186v1) to classify Cancer using a Multi-Instance approach to TCR Repertoires.

For more details regarding this research, please view my dissertation [here](https://google.com).

## Installation

1. Clone this repository
2. Create a Python Environment venv through
``` python3 -m venv $YOUR-VENV-NAME-HERE ```
3. Activate your virtual environment, and run the following command
``` python -m pip install -r scripts/requirements.txt ```
Depending on your Operating System and CUDA requirements, please change ```requirements.txt``` to the following appropriately:

| Operating System | CUDA | Requirements Filename |
|-|-|-|
| Windows | True | ```requirements.txt``` |
| Windows | False | ```requirements-cpuonly.txt``` |
| Linux | True | ```requirements-linux.txt``` |
| Linux | False | ```requirements-linux.txt``` |

## Data

The TRACERx dataset has been used for this task, and it is assumed that you will have access to the Chain Lab RDS.

To pull the data from the Chain Lab RDS, you may run:
```
python loaders/control_scraper.py config_path loaders/config.json -o data
```
Please modify ```rds_mountpoint``` in ```config.json``` to your mountpoint in your computer.  You may leave others as is.

To compress the data (i.e. removing all data other than CDR1, CDR2 and CDR3 sequences), you may run
```
python utils/file-compressor.py
```
where this will generate the folder ```compressed``` under ```data```.

## Model

To download the [TCR-BERT](https://www.biorxiv.org/content/10.1101/2021.11.18.469186v1) pretrained model.  