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
        self.as_dataframe = pd.read_csv(self.data_path)

    def evaluate(self):
        # print(self.predictions)
        labels = self.predictions["label"].tolist()  # 真实标签
        preds = self.predictions["predictions"].tolist()  # 预测标签
        evaluator = Evaluator(labels, preds)
        evaluator.get_scores()
        return evaluator.f_score_nonlit

    def write_file_with_preds(self, target_file):
        self.predictions.to_csv(target_file, sep="\t")
