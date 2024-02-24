import torch
from pathlib import Path
import pandas as pd
import random
import numpy as np


class PatientTCRloader_SCEPTR(torch.utils.data.Dataset):
    def __init__(
        self,
        split = 0.8,
        path="data/sceptr",
        positives=["pbmc_cancer"],
        negatives=["control"],
        shuffle=False
    ):
        super(PatientTCRloader_SCEPTR, self).__init__()

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
        trainidx = np.random.choice(np.arange(len(self.__files)), replace = False, size = int(len(self) * split)).astype(int)
        testidx  = np.setdiff1d(np.arange(len(self.__files)), trainidx).astype(int)
        self.__trainset = [self.__files[i] for i in trainidx]
        self.__testset  = [self.__files[i] for i in testidx]
    
    def set_mode(self, train = True):
        self.training = train

    def __load_files(self, paths):
        return sum([list(i.glob("*")) for i in paths], [])

    def __len__(self):
        if self.training is None:
            return len(self.__files)
        elif self.training:
            return len(self.__trainset)
        else:
            return len(self.__testset)

    def ratio(self, positive=True):
        if positive:
            return sum([i[1] for i in self.__trainset]) / len(self.__trainset)
        else:
            return sum([(1 - i[1]) for i in self.__trainset]) / len(self.__trainset)

    def __getitem__(self, x):
        if self.training is None:
            filepath, label = self.__files[x]
        elif self.training:
            filepath, label = self.__trainset[x]
        else:
            filepath, label = self.__testset[x]

        df = pd.read_csv(filepath, delimiter="\t")
        return [str(filepath), (label, df)]