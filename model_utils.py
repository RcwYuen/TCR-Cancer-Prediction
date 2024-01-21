from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification, AutoModelForMaskedLM
import argparse
import os

class Loader:
    def __init__(self, mlm_only = True, path = None):
        self.__modeltype = "mlm-only" if mlm_only else "ordinary"
        path = "model" if path is None else path
        self.__path = path + "/" + self.__modeltype

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.__path + "/tokenizer")

    def load_model(self):
        if self.__modeltype == "mlm-only":
            return AutoModelForMaskedLM.from_pretrained(self.__path + "/model", output_hidden_states = True)
        else:
            return AutoModelForSequenceClassification.from_pretrained(self.__path + "/model", output_hidden_states = True)

    def load_pipe(self, task):
        return pipeline(task, model = self.__path + "/pipe")
    
    @staticmethod
    def download_models(verbose = False):
        print ("Downloading Models")
        os.system(f"python load_ptm.py" + "--silent" if not verbose else "")
