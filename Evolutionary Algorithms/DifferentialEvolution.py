import numpy as np
import matplotlib.pyplot as plt


def get_three_random(population):
    """
    random choice 3 numbers that different from each other from [0,population)
    :param population: population size
    :return: 3 numbers that different from each other
    """
    x1 = x2 = x3 = 0
    while x1 == x2 or x1 == x3 or x2 == x3:
        x1 = np.random.randint(0, population)
        x2 = np.random.randint(0, population)
        x3 = np.random.randint(0, population)
    return x1, x2, x3


class DifferentialEvolution:
    def __init__(self, func, low, high, dimension=1, population=10, F=1, CR=0.5, epoch=50, verbose=True):
        """
        :param func: objective function
        :param low: range of x
        :param high: range of x
        :param dimension: dimension
        :param population: population size
        :param F: scaling factor
        :param CR: crossover rate
        :param epoch: epoch
        :param verbose: print or not
        """
        self.func = func
        self.low = low
        self.high = high
        self.population = population
        self.dimension = dimension
        self.F = F
        self.CR = CR
        self.epoch = epoch
        self.verbose = verbose
        # initialization
        self.points = low + (high - low) * np.random.rand(population, dimension)
        if self.verbose:
            print(self.points)
        self.optimal_points = self.points.copy()

    def step(self):
        for i in range(self.population):
            # random select 3 species
            x1, x2, x3 = get_three_random(self.population)
            donor = self.points[x1] + self.F * (self.points[x2] - self.points[x3])
            # cross mutation
            new_point = np.zeros_like(self.points[i])
            for j in range(self.dimension):
                rdm = np.random.random()
                if rdm <= self.CR:
                    new_point[j] = donor[j]
                else:
                    new_point[j] = self.points[i][j]
            # Irand
            irand_idx = np.random.randint(0, self.dimension)
            new_point[irand_idx] = donor[irand_idx]
            # selection
            if self.func(new_point) <= self.func(self.points[i]):
                self.points[i] = new_point

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
    # optim = DifferentialEvolution(f1, low=-5.12, high=5.12, dimension=10, population=20, F=1, CR=0.3, epoch=1000, verbose=False)
    # optim = DifferentialEvolution(f2, low=-2.048, high=2.048, dimension=10, population=20, F=1, CR=0.3, epoch=1000, verbose=False)
    # optim = DifferentialEvolution(f3, low=-32.768, high=32.768, dimension=10, population=20, F=1, CR=0.3, epoch=1000, verbose=False)
    optim = DifferentialEvolution(f4, low=-600, high=600, dimension=10, population=20, F=1, CR=0.3, epoch=1000,
                                  verbose=False)
    loss, min_loss = optim.optimize()
    all_loss += [min_loss]
    print("loss:%e" % min_loss)
    print(optim.optimal_points)
    plt.plot(loss)
    plt.show()

# 模拟了一阶梯度
