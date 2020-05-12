import numpy as np
import matplotlib.pyplot as plt
import math


class GeneticAlgorithm:
    def __init__(self, func, low, high, dimension=1, NP=10, NG=50, cross_mutation=0.9, basic_mutation=0.05, precision=3,
                 verbose=True):
        """
        :param func: objective function
        :param low: range of x
        :param high: range of x
        :param dimension: dimension
        :param NP: population size
        :param NG: epoch
        :param cross_mutation: probability of cross mutation
        :param basic_mutation: probability of mutation
        :param precision: precision, 3 means 1e-3
        :param verbose: print or not
        """
        self.func = func
        self.low = low
        self.high = high
        self.dimension = dimension
        self.NP = NP
        self.NG = NG
        self.cross_mutation = cross_mutation
        self.basic_mutation = basic_mutation
        self.verbose = verbose
        self.precision = precision

        # initialize population
        num_range = (high - low) * (10 ** precision)
        self.bits = math.ceil(math.log(num_range) / math.log(2)) + 1  # how many binary bits we need
        # random initialize points
        self.points = low + (np.random.randint(0, 2 ** self.bits, size=(NP, dimension))) / (10 ** precision)
        self.optimal_points = self.points.copy()
        if self.verbose:
            print(self.points)

    def trans_binary(self, x):
        """
        :param x: decimal float
        :return: binary string
        """
        if x < 0:
            return '1' + bin(int(-x * (10 ** self.precision)))[2:].zfill(self.bits - 1)
        else:
            return '0' + bin(int(x * (10 ** self.precision)))[2:].zfill(self.bits - 1)

    def trans_decimal(self, x):
        """
        :param x: binary string
        :return: decimal float
        """
        if x[0] == '0':
            return int(x[1:], 2) / (10 ** self.precision)
        else:
            return -int(x[1:], 2) / (10 ** self.precision)

    def step(self):
        # calculate objective function
        output = np.zeros(self.NP)
        for i, point in enumerate(self.points):
            output[i] = -self.func(point)
        # normalization
        output_max = np.max(output)
        output_min = np.min(output)
        output = (output - output_min) / (output_max - output_min)
        # calculate roulette (use softmax)
        probability = np.exp(output) / np.sum(np.exp(output))
        # calculate next generation
        new_points = np.zeros_like(self.points)
        for i in range(self.NP):
            if self.verbose:
                print(f"No.{i + 1} species")
            new_point = np.zeros(self.dimension)
            for j in range(self.dimension):
                # select father by roulette
                father = self.points[roulette_select(probability)][j]
                # random select mother
                mother = self.points[np.random.randint(0, self.NP)][j]
                # cross mutation
                cross_mutation_flag = False
                cross_mutation_idx = 0
                if np.random.random() < self.cross_mutation:
                    cross_mutation_flag = True
                    cross_mutation_idx = np.random.randint(0, self.bits + 1)  # cut-off point
                    new_point[j] = self.trans_decimal(
                        self.trans_binary(father)[:cross_mutation_idx] + self.trans_binary(mother)[cross_mutation_idx:])
                else:
                    new_point[j] = father
                # mutation
                basic_mutation_flag = False
                basic_mutation_idx = 0
                if np.random.random() < self.basic_mutation:
                    basic_mutation_flag = True
                    basic_mutation_idx = np.random.randint(0, self.bits)  # mutation point
                    new_point_binary = self.trans_binary(new_point[j])
                    if new_point_binary[basic_mutation_idx] == '1':
                        new_point[j] = self.trans_decimal(
                            new_point_binary[:basic_mutation_idx] + '0' + new_point_binary[basic_mutation_idx + 1:])
                    else:
                        new_point[j] = self.trans_decimal(
                            new_point_binary[:basic_mutation_idx] + '1' + new_point_binary[basic_mutation_idx + 1:])
                if self.verbose:
                    print(f"father:{self.trans_binary(father)}({father}),"
                          f"mother:{self.trans_binary(mother)}({mother})", end=",")
                    if cross_mutation_flag:
                        print(f"cross mutation: true, cut-off point:{cross_mutation_idx}", end=",")
                    else:
                        print("cross mutation: false", end=",")
                    if basic_mutation_flag:
                        print(f"mutation: true, mutation point:{basic_mutation_idx}", end=",")
                    else:
                        print("mutation: false", end=",")
                    print(f"generate new species:{self.trans_binary(new_point[j])}({new_point[j]})")
            new_points[i] = new_point
        self.points = new_points

    def optimize(self):
        loss = []
        min_loss = 99999
        for i in range(self.NG):
            # print(f"round {i + 1}:")
            self.step()
            if self.verbose:
                print(f"{self.points}")
            cur_loss = 0
            for point in self.points:
                cur_loss += self.func(point)
            # print(f"loss:{cur_loss}")
            # print("-------------------------")
            loss += [cur_loss]
            if loss[i] < min_loss:
                min_loss = loss[i]
                self.optimal_points = self.points.copy()
        # print("----------done-----------")
        return loss, min_loss


def roulette_select(probability):
    """
    roulette
    :param probability: probability of each choice
    :return: choice(the index of choice)
    """
    rdm = np.random.random()
    sum = 0
    for i, p in enumerate(probability):
        sum += p
        if rdm < sum:
            return i
    return len(probability) - 1


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
    # optim = GeneticAlgorithm(f1, low=-5.12, high=5.12, dimension=d, NP=20, NG=2000, cross_mutation=0.9,
    #                          basic_mutation=0.02, precision=3, verbose=False)
    # optim = GeneticAlgorithm(f2, low=-2.048, high=2.048, dimension=d, NP=20, NG=2000, cross_mutation=0.9,
    #                          basic_mutation=0.02, precision=3, verbose=False)
    # optim = GeneticAlgorithm(f3, low=-32.768, high=32.768, dimension=d, NP=20, NG=2000, cross_mutation=0.9,
    #                          basic_mutation=0.02, precision=3, verbose=False)
    optim = GeneticAlgorithm(f4, low=-600, high=600, dimension=10, NP=20, NG=2000, cross_mutation=0.9,
                             basic_mutation=0.02, precision=3, verbose=False)
    loss, min_loss = optim.optimize()
    all_loss += [min_loss]
    print("loss:%e" % min_loss)
    print(optim.optimal_points)
    plt.plot(loss)
    plt.show()

# 对于第四个函数,GA更有可能卡在local min,只能依靠基本位变异摆脱(而且变异可能又到更大的local min),对GA算法很不友好
# 而且这个basic变异的几率不能太大,不然可能几个位同时变异打破了平衡(朝着更高的loss走了),只能设的很低,让不好的变异位尽快消失,同时又保留好的变异位
