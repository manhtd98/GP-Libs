import numpy as np
from deap import algorithms, base, creator, tools, gp
from .pset import create_pset
from .toolbox import init_toolbox
from .helpers import test_score
from .draw import init_draw
from skmultilearn.dataset import load_dataset


class GPClasification:
    def __init__(
        self, population=512, sample=0.1, epoch=50, alpha=0.8, beta=0.2, hallofframe=1
    ):
        self.population = population
        self.sample = sample
        self.epoch = epoch
        self.toolbox = None
        self.hoft = None
        self.alpha = alpha
        self.beta = beta
        self.hallofframe = hallofframe

    def fit(self, x_train, y_train):
        num_attr = x_train.shape[1]
        self.num_attr = num_attr
        self.pset = create_pset(self.num_attr)
        # print(x_train.shape, y_train.shape)
        x_train = np.hstack([x_train, y_train])
        
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        toolbox = init_toolbox(self.pset, x_train, num_attr, int(self.sample*x_train.shape[0]))
        pop = toolbox.population(n=self.population)
        hof = tools.HallOfFame(self.hallofframe)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
          # pool = multiprocessing.Pool(processes=8)
    # toolbox.register("map", pool.map)
        pop, log = algorithms.eaSimple(
            pop, toolbox, self.alpha, self.beta, self.epoch, stats, halloffame=hof
        )
        self.toolbox = toolbox
        # init_draw(self.toolbox, nlabel)
        self.hoft = hof[0]
        return self

    def predict(self, x_test):
        func = self.toolbox.compile(expr=self.hoft)
        pred = np.array([func(*val) for val in x_test])
        predict = np.where(pred > 0, 1, 0).reshape(-1, 1)
        return predict


# 0.5 -> 0.8
# 0.1 -> 0.2


def train_pipeline(x_train, y_train, num_attr, population=100, sample=200, epoch=50):
    pset = create_pset(num_attr)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    toolboxes = []
    for i in range(y_train.shape[1]):
        print(f"Training tree of class: {i}")
        x_train = np.hstack(
            [x_train, y_train[:, i].reshape(-1, 1)]
        )  # .astype(np.float32)
        toolbox = init_toolbox(pset, x_train, num_attr, sample)
        pop = toolbox.population(n=population)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        pop, log = algorithms.eaSimple(
            pop, toolbox, 0.8, 0.2, epoch, stats, halloffame=hof
        )
        toolboxes.append((pop, toolbox, hof))
    return toolboxes


def evaluation_pipeline(toolboxes, x_test, y_test, key, num_attr):
    print(f"\nStart {key} process:\n")
    # x_test = [x_test[:, i] for i in range(num_attr)]
    preds = []
    for id, (pop, toolbox, hof) in enumerate(toolboxes):
        func = toolbox.compile(expr=hof[0])
        pred = np.array([func(*val) for val in x_test])
        predict = np.where(pred > 0, 1, 0).reshape(-1, 1)
        preds.append(predict)
    preds = np.hstack(preds)
    print(np.sum(preds))
    hamming_loss, f1, acc = test_score(y_test, preds)
    return hamming_loss, f1, acc


if __name__ == "__main__":
    X_train, y_train, feature_names, label_names = load_dataset("emotions", "train")
    X_test, y_test, _, _ = load_dataset("emotions", "test")
    X_train = X_train.toarray()
    y_train = y_train.toarray()

    X_test = X_test.toarray()
    y_test = y_test.toarray()
    num_attr = X_train.shape[1]

    toolboxes = train_pipeline(X_train, y_train, num_attr)
    evaluation_pipeline(toolboxes, X_test, y_test)  # type: ignore
