import numpy as np
import matplotlib.pyplot as plt


class DownhillSimplex:
    def __init__(self, func, low, high, dimension=2, epoch=50, diff=1e-3, verbose=True):
        """
        :param func: objective function
        :param low: range of x
        :param high: range of x
        :param dimension: dimension
        :param epoch: epoch
        :param diff: when the distance of two points is less than diff, then restart some points
        :param verbose: print or not
        """
        self.func = func
        self.low = low
        self.high = high
        self.dimension = dimension
        # initialize points
        self.points = low + (high - low) * np.random.rand(dimension + 1, dimension)
        self.epoch = epoch
        self.diff = diff
        self.optimal_points = self.points.copy()
        self.verbose = verbose

    def sort(self):
        """
        sort points according to their values, ascending order
        """
        self.points = sorted(self.points, key=lambda x: self.func(x))

    def reflection(self, x, para):
        """
        reflection, including expansion and contract
        :param x: point needs reflection
        :param para: the parameter of reflection, 1 means reflection, >1 means expansion, <1 means contract
        :return: point after reflection
        """
        avg_point = (np.sum(self.points, axis=0) - x) / self.dimension
        new_x = avg_point + para * (avg_point - x)
        return new_x

    def shrink(self, x, para=0.5):
        """
        shrink
        :param x: the best point
        :param para: shrink ratio
        """
        self.points = (x - self.points) * para + self.points

    def step(self):
        self.sort()
        worst_x = self.points[-1]
        worst_x_loss = self.func(worst_x)
        reflection_x = self.reflection(worst_x, 1)
        reflection_x_loss = self.func(reflection_x)
        if reflection_x_loss < worst_x_loss:
            # expansion
            expansion_x = self.reflection(worst_x, 2)
            if self.func(expansion_x) < reflection_x_loss:
                # make sure that the x is within the boundary
                expansion_x = np.maximum(expansion_x, self.low)
                expansion_x = np.minimum(expansion_x, self.high)
                op = 1
                self.points[-1] = expansion_x
            else:
                # make sure that the x is within the boundary
                reflection_x = np.maximum(reflection_x, self.low)
                reflection_x = np.minimum(reflection_x, self.high)
                op = 2
                self.points[-1] = reflection_x
        else:
            # contract
            contraction_x = self.reflection(worst_x, 0.5)
            if self.func(contraction_x) < worst_x_loss:
                op = 3
                self.points[-1] = contraction_x
            else:
                # shrink
                op = 4
                self.shrink(self.points[0])
        # calculate loss
        loss = 0
        for point in self.points:
            loss += self.func(point)
        return loss, op

    def optimize(self):
        loss = []
        min_loss = 99999
        restart_cnt = 0
        for i in range(self.epoch):
            cur_loss, op = self.step()
            loss += [cur_loss]
            if self.verbose:
                print("epoch: %d, loss: %d" % (i + 1, loss[i]))
                if op == 1:
                    print("expansion")
                elif op == 2:
                    print("reflection")
                elif op == 3:
                    print("contraction")
                else:
                    print("shrink")
                print(self.points)
                print("-------------------------")
            if loss[i] < min_loss:
                min_loss = loss[i]
                self.optimal_points = self.points.copy()
            self.sort()
            # when the distance of two points is less than diff,
            # then restart some points(prevent falling into local min)
            if self.dimension > 1 and np.sqrt(np.sum((self.points[0] - self.points[-1]) ** 2)) < self.diff:
                best_x = self.points[0:int(self.dimension / 2)]
                self.points = self.low + (self.high - self.low) * np.random.rand(self.dimension + 1, self.dimension)
                self.points[0:int(self.dimension / 2)] = best_x
                restart_cnt += 1
                if self.verbose:
                    print("restart")
        # print("restart: %d" % restart_cnt)
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
    optim = DownhillSimplex(func=f1, low=-5.12, high=5.12, dimension=10, epoch=100, diff=1e-3, verbose=True)
    # optim = DownhillSimplex(func=f2, low=-2.048, high=2.048, dimension=10, epoch=1000, diff=1e-3, verbose=False)
    # optim = DownhillSimplex(func=f3, low=-32.768, high=32.768, dimension=10, epoch=1000, diff=1e-3, verbose=False)
    # optim = DownhillSimplex(func=f4, low=-600, high=600, dimension=10, epoch=1000, diff=1e-3, verbose=False)
    loss, min_loss = optim.optimize()
    all_loss += [min_loss]
    print("loss:%e" % min_loss)
    print(optim.optimal_points)
    plt.plot(loss)
    plt.show()
