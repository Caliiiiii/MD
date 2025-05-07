from add_corpus import Corpus
import argparse
from add_experiments import mBERT_zero
import statistics as stat

def parse_option():
    parser = argparse.ArgumentParser(description="This module carries out cross-lingual metaphor detection.")
    parser.add_argument("-x", "--experiment", type=str, metavar="", required=True, help="zero, few, madx, rf")
    parser.add_argument("-t", "--train_file", type=str, metavar="", required=True, help="File name required of"
                                                                                      "training dataset"
                                                                                      "that contains the "
                                                                                      "required format "
                                                                                      "(see README). File "
                                                                                         "name only, no "
                                                                                         "path required.")
    parser.add_argument("-p", "--predict_file", type=str, metavar="", required=True, help="File name required of"
                                                                                      "predict dataset (test set)"
                                                                                      "that contains the "
                                                                                      "required format "
                                                                                      "(see README).File "
                                                                                         "name only, no "
                                                                                         "path required.")
    parser.add_argument("-l", "--language", type=str, metavar="", required=True, help="Available languages: "
                                                                                      "'ru', 'ge', 'la'.")
def main(args):
    # load data
    file_train = "data/md/" + file_train
    file_test = "data/md/" + file_test
    seeds = [2, 42, 63]

    # read train file:
    corpus_train = Corpus(file_train)
    corpus_train.read()

    # read eval file:
    corpus_predict = Corpus(file_test)
    corpus_predict.read()

    # train and predict:
    results = []
    for seed in seeds:
        if experiment == "zero":
            target_file = "results/mBERT_finetuned"
            corpus_predict.predictions = mBERT_zero(corpus_train,
                                                    corpus_predict, target_file, seed)
        # evaluate:
        corpus_predict.evaluate(seed)
        # print("F1-score (non-literal): ", corpus_predict.evaluate())  
        # results.append(corpus_predict.evaluate())

    # print(results)
    # stand_dev = stat.stdev(results)
    # mean = sum(results)/len(results)
    # print(stand_dev, mean)


if __name__ == '__main__':
    args = parse_option()
    main(args)