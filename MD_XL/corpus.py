"""This file contains the Corpus class that allows
reading and evaluating files as well as writing them."""

import pandas as pd
from evaluate import Evaluator

class Corpus(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.as_dataframe = ""
        self.predictions = ""

    def read(self):
        self.as_dataframe = pd.read_table(self.data_path, sep="\t")
        
    def evaluate(self):
        print(self.predictions)
        labels = self.predictions["label"].tolist()
        preds = self.predictions["predictions"].tolist()
        evaluator = Evaluator(labels, preds)
        evaluator.get_scores()
        return evaluator.f_score_nonlit
    
    def write_file_with_preds(self, target_file):
        self.predictions.to_csv(target_file, sep="\t")