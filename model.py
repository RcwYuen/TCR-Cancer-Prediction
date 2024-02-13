import torch
from sparsemax import Sparsemax
import copy

class unidirectional(torch.nn.Module):
    def __init__(self):
        super(unidirectional, self).__init__()
        self.last_scores = None
        self.last_weights = None

        self.scoring_linear1 = torch.nn.Linear(768, 1)
        self.sparsemax = Sparsemax(dim = 0)
        self.classifying_linear1 = torch.nn.Linear(768, 1)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        self.last_scores = self.scoring_linear1(x)
        # There appears to be a bug with the Sparsemax function.
        # Sparsemax only handles 2-dim tensors, therefore there is a code
        # that does:
        # input = input.transpose(0, self.dim)
        # which means if I have dim = 0, no transpose will happen, therefore I
        # need to transpose the scores first and transpose it back.
        self.last_weights = self.sparsemax(self.last_scores.T).T
        agg_out = torch.sum(self.last_weights * x, dim = 0, keepdim = True)
        result = self.classifying_linear1(agg_out)
        return self.sig(result)

def load_trained(path_to_trained, hypothesised_model):
    model = hypothesised_model() \
        if isinstance(hypothesised_model, type) else \
            copy.deepcopy(hypothesised_model)
    model.load_state_dict(torch.load(path_to_trained))
    return model.cuda() if torch.cuda.is_available() else model

def reset_classifier(model):
    for name, module in model.named_modules():
        if "classifying" in name:
            module.reset_parameters()
            print (f"resetted {name}")