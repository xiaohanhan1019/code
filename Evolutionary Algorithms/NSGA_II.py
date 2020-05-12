import numpy as np
import matplotlib.pyplot as plt
import math
import random


class NSGA_II:
    def __init__(self, funcs, low, high, dimension, population=10, epoch=50, cross_mutation_probability=0.5,
                 mutation_probability=0.1, precision=3, verbose=True):
        """
        :param funcs: objective functions
        :param low: x range
        :param high: x range
        :param dimension: dimension
        :param population: initial population
        :param epoch: epoch
        :param cross_mutation_probability: probability of cross mutation
        :param mutation_probability: probability of mutation
        :param precision: the precision of x, 3 means 1e-3
        :param verbose: print or not
        """
        self.funcs = funcs
        self.low = low
        self.high = high
        self.dimension = dimension
        self.population = population
        self.epoch = epoch
        self.cross_mutation_probability = cross_mutation_probability
        self.mutation_probability = mutation_probability
        self.precision = precision
        self.verbose = verbose
        self.INF = 1e8
        # initialization
        num_range = (high - low) * (10 ** precision)
        self.bits = math.ceil(math.log(num_range) / math.log(2)) + 1  # binary bits, the first bit is sign
        self.points = low + (np.random.randint(0, 2 ** self.bits, size=(population, dimension))) / (10 ** precision)
        self.candidate_points = np.zeros((population * 2, dimension))
        self.optimal_points = self.points.copy()

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

    def cross_mutation(self):
        """
        cross mutation & mutation
        retain the rank 0&1 species and generate new species
        """
        # retain father species
        idx = 0
        for i in range(self.population):
            self.candidate_points[idx] = self.points[i]
            idx += 1
        # sort
        rank = self.non_dominated_sort(self.points)
        rank01_idx = [i for i in range(len(rank)) if rank[i] == 0 or rank[i] == 1]
        # generate n points based on self.points
        for i in range(self.population):
            father = self.points[random.sample(rank01_idx, 1)[0]]  # father is selected from rank 0 or rank 1 point
            mother = self.points[np.random.randint(0, self.population)]
            new_point = father.copy()
            for d in range(self.dimension):
                # cross mutation
                if np.random.random() < self.cross_mutation_probability:
                    cross_mutation_idx = np.random.randint(0, self.bits + 1)
                    new_point[d] = self.trans_decimal(self.trans_binary(father[d])[:cross_mutation_idx]
                                                      + self.trans_binary(mother[d])[cross_mutation_idx:])
                # mutation
                # new_point[d] = self.low + (np.random.randint(0, 2 ** self.bits)) / (10 ** self.precision)
                # one bit mutation
                if np.random.random() < self.mutation_probability:
                    basic_mutation_idx = np.random.randint(0, self.bits)
                    new_point_binary = self.trans_binary(new_point[d])
                    if new_point_binary[basic_mutation_idx] == '1':
                        new_point[d] = self.trans_decimal(
                            new_point_binary[:basic_mutation_idx] + '0' + new_point_binary[basic_mutation_idx + 1:])
                    else:
                        new_point[d] = self.trans_decimal(
                            new_point_binary[:basic_mutation_idx] + '1' + new_point_binary[basic_mutation_idx + 1:])
                if new_point[d] > self.high:
                    new_point[d] = self.high
                if new_point[d] < self.low:
                    new_point[d] = self.low
            self.candidate_points[idx] = new_point
            idx += 1

    def non_dominated_sort(self, points):
        """
        non dominated sort
        :param points: points
        :return: rank
        """
        length = len(points)
        s = [[] for _ in range(length)]  # i dominated the element in s[i]
        n = [0 for _ in range(length)]  # i is dominated by n[i] points
        rank = [-1 for _ in range(length)]  # rank
        f = [[] for _ in range(length)]  # auxiliary array
        for i, p in enumerate(points):
            for j, q in enumerate(points):
                if j <= i:
                    continue
                cnt = 0
                for func in self.funcs:
                    if func(p) < func(q):
                        cnt += 1
                # p dominate q
                if cnt == len(self.funcs):
                    n[j] += 1
                    s[i] += [j]
                # q dominate p
                elif cnt == 0:
                    n[i] += 1
                    s[j] += [i]
            if n[i] == 0:
                rank[i] = 0
                if i not in f[0]:
                    f[0] += [i]
        i = 0
        while f[i]:
            for p in f[i]:
                for q in s[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        if q not in f[i + 1]:
                            f[i + 1] += [q]
            i += 1
        return rank

    def crowding_distance(self, indices):
        """
        calculate crowding distance
        :param indices: indices of points
        :return: sorted indices
        """
        length = len(indices)
        distance = np.zeros(length)
        points = [self.candidate_points[idx] for idx in indices]
        for func in self.funcs:
            values = np.array([func(p) for p in points])
            order = np.argsort(values).tolist()
            min_idx = order.index(0)
            max_idx = order.index(length - 1)
            distance[min_idx] += self.INF
            distance[max_idx] += self.INF
            for i in range(1, length - 1):
                idx = order.index(i)
                prev_idx = order.index(i - 1)
                next_idx = order.index(i + 1)
                distance[idx] += (values[next_idx] - values[prev_idx]) / (np.max(values) - np.min(values) + 1e-3)
        return np.argsort(distance)[::-1].tolist()

    def step(self):
        # generate candidate
        self.cross_mutation()
        # sort
        rank = self.non_dominated_sort(self.candidate_points)
        # choose next generation
        next_generation = np.zeros((self.population, self.dimension))
        idx = 0
        k = 0
        while idx < self.population:
            # choose rank k point
            rank_k = [i for i in range(len(rank)) if rank[i] == k]
            if len(rank_k) < self.population - idx:
                for r in rank_k:
                    next_generation[idx] = self.candidate_points[r]
                    idx += 1
            else:
                rank_k = self.crowding_distance(rank_k)
                for r in rank_k:
                    next_generation[idx] = self.candidate_points[r]
                    idx += 1
                    if idx == self.population:
                        break
        self.points = next_generation

    def optimize(self):
        loss = []
        min_loss = 9999999
        for i in range(self.epoch):
            print(f"round {i + 1}:")
            self.step()
            if self.verbose:
                print(f"{self.points}")
            cur_loss = 0
            for point in self.points:
                for func in self.funcs:
                    cur_loss += func(point)
            print(f"loss:{cur_loss}")
            print("-------------------------")
            loss += [cur_loss]
            if loss[i] < min_loss:
                min_loss = loss[i]
                self.optimal_points = self.points.copy()
        print("----------done-----------")
        return loss, min_loss


"""
The functions 
"""


def f1(x):
    return x[0] ** 2


def f2(x):
    return (x[0] - 2) ** 2


def ZDT1_f1(x):
    return x[0]


def g(x):
    dimension = x.shape[0]
    return 1 + 9 * (np.sum(x) - x[0]) / (dimension - 1)


def ZDT1_f2(x):
    temp = g(x)
    return temp * (1 - np.sqrt(x[0] / temp))


def ZDT2_f2(x):
    temp = g(x)
    return temp * (1 - (x[0] / temp) ** 2)


def ZDT3_f2(x):
    temp = g(x)
    return temp * (1 - np.sqrt(x[0] / temp) - x[0] / temp * np.sin(10 * np.pi * x[0]))


def ZDT4_g(x):
    dimension = x.shape[0]
    temp = np.sum(x ** 2 - 10 * np.cos(4 * np.pi * x)) - x[0] ** 2 + 10 * np.cos(4 * np.pi * x[0])
    return 1 + 10 * (dimension - 1) + temp


def ZDT4_f2(x):
    temp = ZDT4_g(x)
    return temp * (1 - np.sqrt(x[0] / temp))


def ZDT6_f1(x):
    return 1 - np.exp(-4 * x[0]) * (np.sin(6 * np.pi * x[0]) ** 6)


def ZDT6_g(x):
    dimension = x.shape[0]
    return 1 + 9 * ((np.sum(x) - x[0]) / (dimension - 1) ** 0.25)


def ZDT6_f2(x):
    temp = ZDT6_g(x)
    return temp * (1 - (ZDT6_f1(x) / temp) ** 2)


if __name__ == '__main__':
    # SCH
    # funcs = [f1, f2]
    # optim = NSGA_II(funcs=funcs, low=0, high=1, dimension=1, population=100, epoch=10,
    #                 cross_mutation_probability=0.6, mutation_probability=0.2, precision=5, verbose=True)

    # ZDT1
    # funcs = [ZDT1_f1, ZDT1_f2]
    # optim = NSGA_II(funcs=funcs, low=0, high=1, dimension=30, population=50, epoch=200,
    #                 cross_mutation_probability=0.8, mutation_probability=0.05, precision=3, verbose=False)

    # ZDT2
    # funcs = [ZDT1_f1, ZDT2_f2]
    # optim = NSGA_II(funcs=funcs, low=0, high=1, dimension=30, population=50, epoch=200,
    #                 cross_mutation_probability=0.8, mutation_probability=0.05, precision=3, verbose=False)

    # ZDT3
    # funcs = [ZDT1_f1, ZDT3_f2]
    # optim = NSGA_II(funcs=funcs, low=0, high=1, dimension=30, population=50, epoch=200,
    #                 cross_mutation_probability=0.8, mutation_probability=0.05, precision=3, verbose=False)

    # ZDT4 xi:[0,1]
    # funcs = [ZDT1_f1, ZDT4_f2]
    # optim = NSGA_II(funcs=funcs, low=0, high=1, dimension=30, population=50, epoch=200,
    #                 cross_mutation_probability=0.8, mutation_probability=0.05, precision=3, verbose=False)

    # ZDT6
    funcs = [ZDT6_f1, ZDT6_f2]
    optim = NSGA_II(funcs=funcs, low=0, high=1, dimension=30, population=100, epoch=200,
                    cross_mutation_probability=0.8, mutation_probability=0.05, precision=3, verbose=False)

    loss, min_loss = optim.optimize()
    points = optim.optimal_points
    print(points)
    # get rank 0
    rank = optim.non_dominated_sort(points)
    rank0_idx = [i for i in range(len(rank)) if rank[i] == 0]
    rank0_points = [points[i] for i in rank0_idx]
    # draw pic
    v1 = [funcs[0](p) for p in rank0_points]
    v2 = [funcs[1](p) for p in rank0_points]
    plt.xlabel('func1')
    plt.ylabel('func2')
    plt.scatter(v1, v2)
    plt.show()
