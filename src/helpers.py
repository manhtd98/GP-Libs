import random
import numpy as np
from sklearn import metrics

# Define new functions
def protectedDiv(left, right):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x

def test_score(y_test, prediction):
    hamming_loss = metrics.hamming_loss(y_test, prediction)
    f1 = metrics.f1_score(y_test, prediction, average="macro")
    acc = metrics.accuracy_score(y_test, prediction)
    auc_roc = metrics.roc_auc_score(y_test, prediction.toarray(), average="macro")
    jaccard = metrics.jaccard_score(y_test, prediction, average="macro")
    precision_score = metrics.precision_score(y_test, prediction, average="macro")
    recall_score = metrics.recall_score(y_test, prediction, average="macro")
    average_precision_score = metrics.average_precision_score(y_test, prediction.toarray(), average="macro")
    zero_one_loss = metrics.zero_one_loss(y_test, prediction.toarray())
    log_loss = metrics.log_loss(y_test, prediction.toarray())
    label_ranking_loss = metrics.label_ranking_loss(y_test, prediction.toarray())
    res = {
        "hamming_loss": hamming_loss,
        "f1":f1, 
        "accuracy":acc, 
        "auc_roc":auc_roc,
        "jaccard":jaccard,
        "precision_score":precision_score,
        "recall_score":recall_score,
        "avg_precision_score":average_precision_score,
        "zero_one_loss":zero_one_loss,
        "log_loss":log_loss,
        "label_ranking_loss": label_ranking_loss
    }
    print("Result Metrics score: ", res)
    return res


def gen_seed() -> int:
    seed =  random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    return seed
    
