from typing import Callable
import numpy as np
import matplotlib.pyplot as plt


class ABC:
    def __init__(self, objective_func: Callable[[np.ndarray], np.ndarray], nvars: int, lb: float, ub: float,
                 trial_max: int = 10, generations: int = 100, cs: int = 50):
        """
        :param objective_func: Cost function
        :param nvars: Dimensionality of problem
        :param lb: Lower boundry
        :param ub: Upper boundry
        :param trial_max: Number of tries to update before scout phase applies
        :param generations: Number of iterations
        :param cs: Colony Size
        """
        self.cs = cs
        self.nvars = nvars
        self.generations = generations
        self.lb = lb
        self.ub = ub
        self.trial_max = trial_max
        self.optimal_sol = np.array((1, self.nvars))
        self.optimal_fitness = -1
        self.optimality_tracking = np.zeros(shape=(generations, self.nvars))
        self.fitness_tracking = np.full(generations, -1)
        self.cost_tracking = np.zeros(generations)
        self.optimal_cost = None
        self.obj_func = objective_func
        self.trial = np.zeros(self.cs)

    def initialize_colony(self):
        return np.random.uniform(self.lb, self.ub, (self.cs, self.nvars))

    def initialize_individual(self):
        return np.random.uniform(self.lb, self.ub, (1, self.nvars))

    def get_best_sol(self, col):
        cost = self.compute_cost(col)
        fitness = self.compute_fitness(cost)

        max_fitness = np.argmax(fitness)
        if fitness[max_fitness] > self.optimal_fitness:
            self.optimal_sol = np.copy(col[max_fitness])
            self.optimal_fitness = np.copy(fitness[max_fitness])
            self.optimal_cost = np.copy(cost[max_fitness])

    def add_solution_to_tracking_array(self, i):
        self.fitness_tracking[i] = self.optimal_fitness
        self.optimality_tracking[i] = self.optimal_sol
        self.cost_tracking[i] = self.optimal_cost

    def compute_cost(self, bees: np.ndarray) -> np.ndarray:
        if bees.ndim == 1:
            return self.obj_func(bees)
        else:
            return np.array(list(map(self.obj_func, bees)))

    @staticmethod
    def compute_fitness(cost):
        '''
        Static method that computes fitness value for given cost
        :type cost: float or np.ndarray
        :rtype: float or np.ndarray
        '''
        if type(cost) is np.ndarray:
            fitness = np.zeros(cost.shape)
            fitness[cost >= 0] = 1 / (1 + cost[cost >= 0])
            fitness[cost < 0] = 1 + np.abs(cost[cost < 0])
        else:
            fitness = 1 / (1 + cost) if cost >= 0 else 1 + np.abs(cost)
        return fitness

    def check_boundries(self, x: float) -> float:
        """
        Method that checks if a new location is within boundries
        :param x: new location
        :return: float
        """
        if x < self.lb:
            return self.lb
        elif x > self.ub:
            return self.ub
        return x

    def compute_prob(self, col: np.ndarray) -> np.ndarray:
        costs = self.compute_cost(col)
        fitness = self.compute_fitness(costs)

        prob = fitness / np.sum(fitness)
        return prob

    def roulette_wheel_selection(self, p: np.ndarray) -> int:
        '''
        Method that chooses onlooker with given probability
        :param p: probabilities
        :return: index of chosen individual
        '''

        # Get random number betwenn 0 and 1
        rand = np.random.uniform()

        subs = p - rand

        # For each element in subs that is < 0, set to nan
        subs[subs < 0] = np.nan

        # if rand number is greater than each probability, each element in subs is nan
        # so the procedure is repeated until there is an element with values != nan
        while np.isnan(subs).all():
            rand = np.random.uniform()
            subs = p - rand
            subs[subs < 0] = np.nan

        # return index of minimum, which is the closest probability on the right side from rand
        return np.nanargmin(subs)

    def update(self, colony: np.ndarray, colony_fitness: np.ndarray, prob=None):
        '''
        Method that is used in employed bees phase and onlooker phase.
        Generally it searches new location
        Depends on prob param, it may be employed bee or onlooker.
        :param colony: colony of bees
        :param colony_fitness: fitness of each bee location
        :param prob: probability that onlooker will go to place which is descibed by employeed bee which performs a dance
        '''
        n_new_bees = colony.shape[0]
        newbees = np.copy(colony)
        for i in range(n_new_bees):
            idx = i
            if prob is not None:
                idx = self.roulette_wheel_selection(prob)
            # Choose k randomly, not equal to i
            k = np.random.choice(np.delete(np.arange(n_new_bees), i))

            # Choose random variable
            j = np.random.choice(self.nvars)

            phi = np.random.uniform(-1, 1)

            # Calculating new position
            x = colony[idx, j] + phi * (colony[idx, j] - colony[k, j])

            # Set bee's position to x or to the boundry if x exceeds it
            newbees[idx, j] = self.check_boundries(x)

        newbees_costs = self.compute_cost(newbees)
        newbees_fitness = self.compute_fitness(newbees_costs)

        # For the colony choose better positions
        # Set to new bee where new bee fitness is greater than current bee fitness
        colony[newbees_fitness > colony_fitness] = newbees[newbees_fitness > colony_fitness]

        # Increment trial for bees that haven't changed position.
        self.trial[newbees_fitness < colony_fitness] += 1

        # Reset tial variable for new bees.
        self.trial[newbees_fitness > colony_fitness] = 0

    def employed_bee_phase(self, colony, colony_fitness):
        self.update(colony, colony_fitness)

    def onlooker_be_phase(self, col, prob):
        cost = self.compute_cost(col)
        fitness = self.compute_fitness(cost)
        self.update(col, fitness, prob)

    def scout_phase(self, col):
        for i in range(col.shape[0]):
            # Create new scout for bees that trial exceeds its maximum value
            if self.trial_max < self.trial[i]:
                col[i] = self.initialize_individual()
                self.trial[i] = 0

    def optimize(self) -> [np.ndarray, np.ndarray]:

        colony = self.initialize_colony()
        for i in range(self.generations):
            colony_cost = self.compute_cost(colony)
            colony_fitness = self.compute_fitness(colony_cost)

            self.employed_bee_phase(colony, colony_fitness)

            # onlooker bee phase
            prob = self.compute_prob(colony)

            self.onlooker_be_phase(colony, prob)

            # scout_phase
            self.scout_phase(colony)

            # Change optimal solution
            self.get_best_sol(colony)

            # Add best solution so far to the optimality tracking array.
            self.add_solution_to_tracking_array(i)
        return self.cost_tracking, self.optimality_tracking


def perform_n_runs(n: int, func, nvars: int, lb: float, ub: float, generatios: int, verbose: bool=False) -> [list, list]:
    costs = []
    sols = np.zeros((n, generatios, nvars))
    idx_of_final_sol = []

    for i in range(n):
        cost, sol = ABC(func, nvars, lb, ub, generations=generatios).optimize()

        tmp_sol = sol[-1]

        # Get index of first occurrence of final solution
        idx_of_final_sol.append(np.where(np.array(sol == tmp_sol).all(axis=1))[0][0])

        costs.append(cost[:idx_of_final_sol[i]])
        sols[i] = sol

        if verbose:
            print('Execution: {nr}\nMinimum at: {sol}\nCost: {cost}\nFinal solution at {idx} generation\n'.format(nr=i+1,
                                                                                                             sol=tmp_sol,
                                                                                                 cost=cost[-1],
                                                                                             idx=idx_of_final_sol[i]))

    return costs, sols, idx_of_final_sol


def results_plt(costs: list, sols: np.ndarray, idx: list):
    '''
    Function that creates plot and table with perfomance of each algorithm execution
    :param costs: 2 dimensional list with costs tracking for each algorithm execution
    :param sols: 3 dimensional array with solutions for each algorithm execution
    :param idx: index of generation which attains a minimum
    '''
    if sols.shape[2] != 2:
        return

    costs = costs[:5]
    sols = sols[:5]
    idx = idx[:5]


    sol = sols[:, -1, :]
    best_costs = [cost[-1] for cost in costs]

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

    x, y = sol[:, 0], sol[:, 1]
    cols = [['{:.2e}'.format(x[i]), '{:.2e}'.format(y[i]), idx[i], '{:.6e}'.format(best_costs[i])] for i in
            range(len(idx))]
    colLabels = ['x', 'y', 'Generations needed', 'Cost']

    row_idx = ['{i: ^10}'.format(i=i) for i in range(len(idx))]

    tab0 = ax[1].table(cellText=cols, cellLoc='center', colLabels=colLabels, rowLabels=row_idx,
                       rowColours=['orange'] * len(row_idx),
                       colColours=['palegreen'] * len(colLabels), bbox=[0, -.2, 1, 1])
    tab0.auto_set_font_size(False)
    tab0.set_fontsize(9)
    ax[1].axis('tight')
    ax[1].axis('off')

    line = [None] * len(idx)
    for i, cost in enumerate(costs):
        line[i], = ax[0].plot(np.arange(idx[i]), cost[:idx[i]])
        line[i].set_label(str(i))

    ax[0].legend()

    ax[0].set_xticks([x for x in np.arange(0, max(idx), max(idx) // 10)])
    ax[0].set_xlabel('Generations')
    ax[0].set_ylabel('Cost')
    ax[0].grid()
    ax[0].set_title('Cost plot after n generations for first 5 executions')

    file = 'ABCiterations_results.png'
    plt.savefig(file)
    print('Plot saved in file \"{}\"\n'.format(file))


def stat(idx, costs):
    final_costs = np.array([cost[-1] for cost in costs])
    final_cost_mean = np.mean(final_costs)
    final_cost_std = np.std(final_costs)

    idx_mean = np.mean(idx)
    idx_std = np.std(idx)

    print('Final cost mean: {cost_mean:.4e}\nFinal cost standard deviation: {cost_std:.4e}\n'
          '\nMean of generations that reached minimum: {idx_mean}\nStandard deviation fo generations that reached'
          ' minimum: {index_std:.4f}'.format(cost_mean=final_cost_mean, cost_std=final_cost_std,
                                         idx_mean=idx_mean, index_std=idx_std))


def FunctionToMinimalize(x):
    return -np.floor(np.prod(x)) + x @ x.T


if __name__ == '__main__':
    costs, sols, idx = perform_n_runs(20, FunctionToMinimalize, 2, -50, 50, 50, True)
    results_plt(costs, sols, idx)
    stat(idx, costs)

