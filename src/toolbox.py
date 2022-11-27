import multiprocessing
import random
import numpy as np
from deap import base, creator, tools, gp
from sklearn.metrics import f1_score


def init_toolbox(pset, samples, num_attr, sample_num):
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    # toolbox.register("compile", gp.compile, pset=pset)

    def evalMultiplexer(individual, sample=sample_num):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual)
        spam_samp = random.sample(range(samples.shape[0]), sample)
        # Evaluate the sum of correctly identified
        inputs = samples[spam_samp, :num_attr]
        outputs = samples[spam_samp, num_attr]
        preds = np.array([func(*inputs[i]) for i in range(sample)])
        preds = np.where(preds > 0, 1, 0)
        result = -f1_score(preds, outputs, average='micro')  # weighted
        return (result,)

    toolbox.register("evaluate", evalMultiplexer)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    # Process Pool of 4 workers
    # pool = multiprocessing.Pool(processes=8)
    # toolbox.register("map", pool.map)
    return toolbox
