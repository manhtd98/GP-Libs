import json
import time
from skmultilearn.dataset import load_dataset
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from src.helpers import test_score
from src.main import GPClasification
from src.helpers import gen_seed
from src.pset import create_pset
import numpy as np
import os
datasets = (
    "birds",
    "emotions",
    "enron",
    "genbase",
    "medical",
    "yeast",
    "scene",
    "rcv1subset1",
    "tmc2007_500",
)


def load_datasets(dataset_name):
    X_train, y_train, feature_names, label_names = load_dataset(
        dataset_name, "train")
    X_test, y_test, _, _ = load_dataset(dataset_name, "test")
    X_train = X_train.toarray()
    y_train = y_train.toarray()  # [:, 0].reshape(-1, 1)

    X_test = X_test.toarray()
    y_test = y_test.toarray()  # [:, 0].reshape(-1, 1)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    return X_train, y_train, X_test, y_test


def run_experiments(dataname, out_file="tmp.json"):
    X_train, y_train, X_test, y_test = load_datasets(dataname)
    start_time = time.time()
    classifiers = []
    predict_labels = []
    for i in range(y_train.shape[1]):
        classifier = GPClasification()
        classifier.fit(X_train, y_train[:, i].reshape(-1, 1), "ex1-binary-"+str(i))
        predict_labels.append(classifier.predict(X_train))
        classifiers.append(classifier)
    predict_labels = np.hstack(predict_labels)
    
    gp_relation = []
    predict_labels_gp = []
    for i in range(y_train.shape[1]):
        classifier = GPClasification()
        classifier.fit(
            predict_labels, y_train[:, i].reshape(-1, 1), "ex1-relation-"+str(i))
        predict_labels_gp.append(classifier.predict(predict_labels))
        gp_relation.append(classifier)
    train_time = time.time() - start_time
    predict_labels_gp = np.hstack(predict_labels_gp)
    res_train = test_score(y_train, predict_labels)
    res_train = test_score(y_train, predict_labels_gp)

    # TEST LOOP
    predict_labels = []
    for classifier in classifiers:
        predict_labels.append(classifier.predict(X_test))
    predict_labels = np.hstack(predict_labels)
    predict_labels_gp = []
    for classifier in gp_relation:
        predict_labels_gp.append(classifier.predict(predict_labels))
    predict_labels_gp = np.hstack(predict_labels_gp)
    res_train = test_score(y_test, predict_labels)
    res_test = test_score(y_test, predict_labels_gp)
    res = {
        "result_train": res_train,
        "result_test": res_test,
        "training sample": X_train.shape[0],
        "test sample": X_test.shape[0],
        "training time": train_time,
    }
    dir_name = "/".join(out_file.split("/")[:-1])
    if not os.path.exists(dir_name):
      os.makedirs(dir_name)
    with open(out_file, "w") as f:
        json.dump(res, f)
    return res
def run_gp(i, dataname):
    gen_seed()
    output_name = f"results/{dataname}/gp/{i}.json"
    run_experiments(dataname=dataname, out_file=output_name)
if __name__ == "__main__":
    run_gp(1, "emotions")
    # number_of_cpu = joblib.cpu_count()
    # for data in datasets:
    #       logger.info(f"RUNNING EXPERIMENCE {data}: {number_of_cpu} CORES")
    #       delayed_funcs = [delayed(run_gp)(i, data) for i in range(30)]
    #       parallel_pool = Parallel(n_jobs=number_of_cpu)
    #       parallel_pool(delayed_funcs)
