import torch
from sparsemax import Sparsemax
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

class classifier(torch.nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.last_scores = None
        self.last_weights = None

        self.scoring_linear1 = torch.nn.Linear(768, 512)
        self.scoring_linear2 = torch.nn.Linear(512, 128)
        self.scoring_linear3 = torch.nn.Linear(128, 1)
        
        self.relu = torch.nn.ReLU()
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

df = pd.read_parquet("dcr_beta_LTX203_LN1_B.parquet")
embeddings = torch.from_numpy(df.values).to(torch.float32).cuda()
ytrue = torch.ones((1, 1)).cuda()
model = classifier().cuda()
criterion = torch.nn.BCELoss()
optim = torch.optim.Adam(model.parameters())

train_loss = []

for i in tqdm(range(100)):
    ypred = model(embeddings)
    loss = criterion(ytrue, ypred)

    train_loss.append(loss.data.tolist())
    optim.zero_grad()
    loss.backward()
    optim.step()
