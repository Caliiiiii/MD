"""
This module contains evaluation functions.
"""


# 通过遍历所有预测值，统计预测正确的数量 true。
# 输出：返回准确率，即预测正确的样本数除以总样本数。
def compute_acc(preds, labels):
    true = 0
    for index in range(len(preds)):
        if preds[index] == labels[index]:
            true += 1
    return true / len(preds)


def tp_fp_fn(predictions: list, labels: list, literal_nonliteral: int) -> int:
    """This function calculates true positives,
    false positives and false negatives for a given gender.
    """
    tp = 0
    fp = 0
    fn = 0
    for index in range(len(predictions)):
        if labels[index] == literal_nonliteral:
            if labels[index] == predictions[index]:
                tp += 1
            else:
                fn += 1
        elif labels[index] != literal_nonliteral:
            if predictions[index] == literal_nonliteral:
                fp += 1
    return tp, fp, fn


def precision_recall(tp: int, fp: int, fn: int) -> float:
    """This function calculates precision and recall
    out of true positives, false positives and false negatives.
    """
    precision = 0
    recall = 0
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    print("precision: " + str(precision) + " , recall: " + str(recall))
    return precision, recall


def compute_f_score(precision: float, recall: float, alpha: float) -> float:
    """This function computes f-score out of precision and recall.
    Alpha is automatically given in corpus as 0.5 (harmonic mean),
    but can be changed.
    """
    if precision == 0 or recall == 0:  # if precision or recall are 0,
        # the metric cannot be used and is therefore 0
        return 0
    f_score = 1 / ((alpha * (1 / precision)) + ((1 - alpha) * (1 / recall)))
    return f_score


class Evaluator(object):
    """This class is used to perform evaluation on the user-level
    for perceptron.
    """

    def __init__(self, labels, predictions, alpha=0.5):
        self.labels = labels
        self.predictions = predictions
        self.f_score_lit = 0
        self.f_score_nonlit = 0
        self.accuracy = 0
        self.alpha = alpha

    # 作者修改代码前的
    # def get_scores(self):  # calculates the f-scores
    #     # 为每个类别计算精确率和召回率
    #     tp_f, fp_f, fn_f = tp_fp_fn(self.predictions, self.labels, 0)
    #     tp_m, fp_m, fn_m = tp_fp_fn(self.predictions, self.labels, 1)
    #     precision_m, recall_m = precision_recall(tp_m, fp_m, fn_m)
    #     precision_f, recall_f = precision_recall(tp_f, fp_f, fn_f)
    #     self.f_score_lit = compute_f_score(precision_m, recall_m, self.alpha)
    #     self.f_score_nonlit = compute_f_score(precision_f, recall_f, self.alpha)
    #     self.accuracy = compute_acc(self.predictions, self.labels)

    # 作者两个月前更新了代码仓库
    def get_scores(self):
        tp_nonlit, fp_nonlit, fn_nonlit = tp_fp_fn(self.predictions, self.labels, 1)
        tp_lit, fp_lit, fn_lit = tp_fp_fn(self.predictions, self.labels, 0)
        precision_nonlit, recall_nonlit = precision_recall(tp_nonlit, fp_nonlit, fn_nonlit)
        precision_lit, recall_lit = precision_recall(tp_lit, fp_lit, fn_lit)
        self.f_score_lit = compute_f_score(precision_lit, recall_lit, self.alpha)
        self.f_score_nonlit = compute_f_score(precision_nonlit, recall_nonlit, self.alpha)
        self.accuracy = compute_acc(self.predictions, self.labels)

    def __str__(self):
        return str(self.f_score_nonlit)
