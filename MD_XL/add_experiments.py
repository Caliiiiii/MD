"""This file contains the experiments zero-shot classification
 with mBERT, few-shot classification with mBERT, MAD-X and random
 forest classification."""


from add_train import TrainerMbert, set_seed
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from feature_generator_rf import *
from helper_functions_classi import replace_foreign_with_en
from numpy import *


def mBERT_zero(dataframe_train, dataframe_predict, target_file, seed):
    trainer_1 =  TrainerMbert(dataframe_train=dataframe_train.as_dataframe,
                              dataframe_test=dataframe_predict.as_dataframe,
                              target_file=target_file, seed=seed)

    #train:
    trainer_1.train() # 训练集

    #predict:
    predictions = trainer_1.predict() 
    return predictions