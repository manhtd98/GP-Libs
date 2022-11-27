import itertools
import random
import numpy as np
from .helpers import protectedDiv
from deap import gp
import operator


def create_pset(num_attr):
    pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, num_attr), float, "IN")
    # boolean operators
    # pset.addPrimitive(operator.and_, [bool, bool], bool)
    # pset.addPrimitive(operator.or_, [bool, bool], bool)
    # pset.addPrimitive(operator.not_, [bool], bool)
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(protectedDiv, [float, float], float)
    # # logic operators
    # # Define a new if-then-else function
    # def if_then_else(input, output1, output2):
    #     if input:
    #         return output1
    #     else:
    #         return output2

    # pset.addPrimitive(operator.lt, [float, float], bool)
    # pset.addPrimitive(operator.eq, [float, float], bool)
    # pset.addPrimitive(if_then_else, [bool, float, float], float)
    pset.addEphemeralConstant("rand"+str(random.randint(1, 100000)), \
        lambda: random.random() * 100, float)
    # pset.addEphemeralConstant("rand101", lambda: random.randint(0, 1), float)
    # pset.addTerminal(False, bool)
    # pset.addTerminal(True, bool)

    return pset
