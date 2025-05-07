"""This file contains the experiments zero-shot classification
 with mBERT, few-shot classification with mBERT, MAD-X and random
 forest classification."""


# from ADtrain import TrainerMbert, set_seed # new add
from train0310 import TrainerMbert,set_seed
from sklearn.model_selection import train_test_split
from helper_functions_classi import replace_foreign_with_en
from numpy import *


def mBERT_zero(dataframe_train, dataframe_predict, target_file, seed):
    trainer_1 =  TrainerMbert(dataframe_train=dataframe_train.as_dataframe,
                              dataframe_test=dataframe_predict.as_dataframe,
                              target_file=target_file, seed=seed)

    #train:
    trainer_1.train() 

    #predict:
    predictions = trainer_1.predict() 
    return predictions


"""Checkpoint can either be given or retrieved by mBERT_train()."""
def mBERT_few(checkpoint, dataframe_train_2, seed):
    # get 20 instances from test dataset for second fine-tuning
    # (use the rest of the test set for evaluation):
    train, test = train_test_split(dataframe_train_2.as_dataframe, train_size=500)
    trainer_2 = TrainerMbert(dataframe_train=train,
                             dataframe_test=test,
                             target_file="results/fine-tuned_twice", seed=seed)

    # replace pretrained mBERT by mBERT fine-tuned on source language material:
    trainer_2.checkpoint = checkpoint

    #train:
    trainer_2.train()

    #predict:
    predictions = trainer_2.predict()
    return predictions


def mBERT_MADX(dataframe_train, dataframe_predict, target_file, language, path_task_adapter,seed):
    trainer = TrainerMbert(dataframe_train=dataframe_train.as_dataframe,
                           dataframe_test=dataframe_predict.as_dataframe,
                           target_file=target_file, seed=seed)
    # return trainer.mad_x(language, path_task_adapter)
    return trainer.train_madx() 

