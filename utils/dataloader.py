import torch
from pathlib import Path
import pandas as pd
import random
import numpy as np


class PatientTCRloader(torch.utils.data.Dataset):
    def __init__(
        self,
        split = 0.8,
        path="data/files",
        positives=["lung_cancer", "pbmc_cancer"],
        negatives=["control"],
        cdr1=False,
        cdr2=False,
        shuffle=False
    ):
        super(PatientTCRloader, self).__init__()
        self.__columns_needed = ["junction_aa"]
        self.__cdr3only = not(cdr1 or cdr2)
        if cdr1:
            self.__columns_needed.append("cdr1_aa")
        if cdr2:
            self.__columns_needed.append("cdr2_aa")

        self.__positive_files = self.__load_files(
            [Path.cwd() / path / i for i in positives]
        )
        self.__negative_files = self.__load_files(
            [Path.cwd() / path / i for i in negatives]
        )
        self.__files = [(i, 1) for i in self.__positive_files] + \
                       [(i, 0) for i in self.__negative_files]
        if shuffle:
            random.shuffle(self.__files)
        self.training = None
        trainidx = np.random.choice(np.arange(len(self)), size = int(len(self) * split)).astype(int)
        print (trainidx)
        testidx  = np.setdiff1d(np.arange(len(self)), trainidx).astype(int)
        self.__trainset = self.__files[trainidx]
        self.__testset  = self.__files(testidx)
    
    def set_mode(self, train = True):
        self.training = train

    def __load_files(self, paths):
        files = sum([list(i.glob("*")) for i in paths], [])
        files = [i for i in files if self.__filetype(i) != ".cdr3" and self.__cdr3only]
        return files

    def __len__(self):
        if self.training is None:
            return len(self.__files)
        elif self.training:
            return len(self.__trainset)
        else:
            return len(self.__testset)

    def ratio(self, positive=True):
        return (
            len(self.__positive_files) / len(self)
            if positive
            else len(self.__negative_files) / len(self)
        )

    def __filetype(self, filepath):
        return filepath.suffix[1::]

    def __getitem__(self, x):
        if self.training is None:
            filepath, label = self.__files[x]
        elif self.training:
            filepath, label = self.__trainset[x]
        else:
            filepath, label = self.__testset[x]

        if self.__filetype(filepath) == "tsv":
            df = pd.read_csv(filepath, delimiter="\t")[self.__columns_needed]
        else:
            df = pd.read_csv(filepath, header=None, usecols=[0])

        df = df.dropna(axis=0, how="all")
        df = df.map(lambda x: " ".join(list(x)))
        df = df.apply(lambda x: "|".join(x), axis = 1)
        return [str(filepath), (label, df.values.tolist())]


class TCRloader(torch.utils.data.Dataset):
    def __init__(self, label, repertoire):
        super(TCRloader, self).__init__()
        self.__repertoire = repertoire
        self.__label = label
        self.data = torch.utils.data.Subset(self, np.arange(len(self)))

    def __len__(self):
        return len(self.__repertoire)

    def __getitem__(self, x):
        return self.__repertoire[x]