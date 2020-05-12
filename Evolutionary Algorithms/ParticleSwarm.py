import numpy as np
import matplotlib.pyplot as plt


class ParticleSwarm:
    def __init__(self, func, low, high, dimension=1, population=10, c1=0.5, c2=0.5, w=0.5, epoch=50, verbose=True):
        """
        :param func: objective function
        :param low: range of x
        :param high: range of x
        :param dimension: dimension
        :param population: population size
        :param c1: weight of local information
        :param c2: weight of global information
        :param w:
        :param epoch: epoch
        :param verbose: print of not
        """
        self.func = func
        self.low = low
        self.high = high
        self.population = population
        self.dimension = dimension
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.epoch = epoch
        self.verbose = verbose
        # initialization
        self.points = low + (high - low) * np.random.rand(population, dimension)
        if self.verbose:
            print(self.points)
        self.optimal_points = self.points.copy()
        # history best(pbest)
        self.points_history = self.points.copy()
        # initialize path direction v [-1,1]
        self.points_v = (-1 + np.random.rand(population, dimension) * 2) * c1

    def step(self):
        # gbest
        global_best = self.points[0]
        for point in self.points:
            if self.func(point) < self.func(global_best):
                global_best = point
        # update each point
        for i, _ in enumerate(self.points):
            # update path direction v
            self.points_v[i] = self.w * self.points_v[i] \
                               + self.c1 * np.random.random() * (self.points_history[i] - self.points[i]) \
                               + self.c2 * np.random.random() * (global_best - self.points[i])
            # make sure that the v won't be too big
            for j, _ in enumerate(self.points_v[i]):
                if self.points_v[i][j] > (self.high - self.low) / 4:
                    self.points_v[i][j] = (self.high - self.low) / 4
                if self.points_v[i][j] < -(self.high - self.low) / 4:
                    self.points_v[i][j] = -(self.high - self.low) / 4
            # update
            self.points[i] = self.points[i] + self.points_v[i]
            # make sure that the x is within the boundary
            for j, _ in enumerate(self.points[i]):
                if self.points[i][j] > self.high:
                    self.points[i][j] = self.high
                if self.points[i][j] < self.low:
                    self.points[i][j] = self.low
            # update history point
            if self.func(self.points[i]) < self.func(self.points_history[i]):
                self.points_history[i] = self.points[i]

    def optimize(self):
        loss = []
        min_loss = 99999
        for i in range(self.epoch):
            print(f"round {i + 1}:")
            self.step()
            if self.verbose:
                print(f"{self.points}")
            cur_loss = 0
            for point in self.points:
                cur_loss += self.func(point)
            print(f"loss:{cur_loss}")
            print("-------------------------")
            loss += [cur_loss]
            if loss[i] < min_loss:
                min_loss = loss[i]
                self.optimal_points = self.points.copy()
        print("----------done-----------")
        return loss, min_loss


# Eillipsoid Problem
def f1(x):
    dimension = x.shape[0]
    idx = np.array(range(dimension)) + 1
    return np.sum(idx * (x ** 2))


# Rosenbrock Problem
def f2(x):
    dimension = x.shape[0]
    sum = 0
    for i in range(dimension - 1):
        sum += 100 * (x[i + 1] - x[i] * x[i]) ** 2 + (1 - x[i]) ** 2
    return sum


# Ackley Problem
def f3(x):
    dimension = x.shape[0]
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / dimension)) - np.exp(sum2 / dimension) + np.exp(1) + 20


# Griewank Problem
def f4(x):
    dimension = x.shape[0]
    sum1 = np.sum(x ** 2) / 4000
    sum2 = np.prod(np.cos(x) / np.sqrt(np.arange(1, dimension + 1)))
    return 1 + sum1 - sum2


if __name__ == '__main__':
    all_loss = []
    optim = ParticleSwarm(f1, low=-5.12, high=5.12, dimension=10, population=50, c1=1, c2=1.2, w=0.8, epoch=1000,
                          verbose=False)
    # optim = ParticleSwarm(f2, low=-2.048, high=2.048, dimension=10, population=50, c1=1, c2=1, w=0.8, epoch=1000,
    #                       verbose=False)
    # optim = ParticleSwarm(f3, low=-32.768, high=32.768, dimension=10, population=50, c1=1, c2=1, w=0.8, epoch=1000,
    #                       verbose=False)
    # optim = ParticleSwarm(f4, low=-600, high=600, dimension=10, population=50, c1=1, c2=1, w=0.8, epoch=1000,
    #                       verbose=False)
    loss, min_loss = optim.optimize()
    all_loss += [min_loss]
    print("loss:%e" % min_loss)
    print(optim.optimal_points)
    plt.plot(loss)
    plt.show()
