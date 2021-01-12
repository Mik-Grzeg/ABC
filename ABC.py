import numpy as np
from typing import Callable


class ABC:
    def __init__(self, objective_func: Callable[[np.ndarray], np.ndarray], nvars: int, lb: float, ub: float,
                 generations: int = 1000, cs: int = 50):
        """
        :param objective_func: Cost function
        :param nvars: Dimensionality of problem
        :param lb: Lower boundry
        :param ub: Upper boundry
        :param generations: Number of iterations
        :param cs: Colony Size
        """
        self.cs = cs
        self.nvars = nvars
        self.lb = lb
        self.ub = ub
        self.optimality_tracking = np.zeros(shape=(generations,1))
        self.obj_func = objective_func

    def initialize_colony(self):



Sphere = lambda x: x @ x.T
abc = ABC.ini