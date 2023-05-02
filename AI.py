import math
from random import random
from PIL import Image
import os

PIXELS = 40
LEARNING_RATE = 0.05


def s(x): return 0 if x < -0.5 else (x + 0.5 if x < 0.5 else 1)


def ds(x):
    return 1 if 0 < x < 1 else 0


class Neuron:
    def __init__(self, layer_index, weights=None, bias=None):
        self.weights = [] if weights is None else weights
        self.bias = (2 * random() - 1) if bias is None else bias
        self.network = None
        self.layer_index = layer_index
        self.activation = 0
        self.out = 0
        self.delta = 0
        self.input = []

    def initialize(self, network):
        self.network = network
        # self.weights = [0]*len(self.network[self.layer_index - 1])
        self.weights = [2 * random() - 1 for _ in range(self.network.sizes[self.layer_index - 1])]

    def activate(self):
        self.input = self.network[0] if self.layer_index == 1 \
                 else [neuron.out for neuron in self.network[self.layer_index - 1]]
        # else list(map(lambda neuron: neuron.out, self.network[self.layer_index - 1]))
        self.activation = self.bias
        for n, num in enumerate(self.input):
            self.activation += self.weights[n] * num
        self.out = s(self.activation)
        return self.activation


class Network(list):
    def __init__(self):
        super(Network, self).__init__()
        self.size = 0
        self.sizes = []

    def initialize(self, num_input, num_hidden, neu_num, num_out):
        self.extend([[0] * num_input,
                     *[[Neuron(n + 1) for _ in range(neu_num)] for n in range(num_hidden)],
                    [Neuron(num_hidden + 1) for _ in range(num_out)]])
        self.size = len(self)
        self.sizes = [len(layer) for layer in self]
        for i in range(1, num_hidden + 2):
            for neuron in self[i]:
                neuron.initialize(self)

    def forward(self, input_):
        for n, layer in enumerate(self):
            if n == 0:
                for i in range(self.sizes[n]):
                    layer[i] = input_[i]
            else:
                for neuron in layer:
                    neuron.activate()

    def backpropagation(self, target):
        for i in reversed(range(1, self.size)):
            layer = self[i]
            errors = []
            if i != self.size - 1:
                for j in range(self.sizes[i]):
                    error = 0.0
                    for neuron in self[i + 1]:
                        error += (neuron.weights[j] * neuron.delta)
                    errors.append(error)
            else:
                for j, neuron in enumerate(layer):
                    errors.append(neuron.out - target[j])
            for j, neuron in enumerate(layer):
                neuron.delta = errors[j] * ds(neuron.out)
        for j in range(1, self.size):
            inputs = self[j][0].input
            for neuron in self[j]:
                for n in range(self.sizes[j - 1]):
                    neuron.weights[n] -= LEARNING_RATE * neuron.delta * inputs[n]
                neuron.bias -= LEARNING_RATE * neuron.delta

    def get_output(self, layer_index=-1):
        return [neuron.out for neuron in self[layer_index]]


def main():
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    targets = [[0], [1], [1], [0]]

    net = Network()
    net.initialize(2, 3, 5, 1)
    for _ in range(50000):
        for n in range(4):
            net.forward(inputs[n])
            net.backpropagation(targets[n])

    for inp in inputs:
        net.forward(inp)
        print(net.get_output())

    # for i, layer in enumerate(net[1:]):
    #     for neuron in layer:
    #         print(i, neuron.weights)


if __name__ == '__main__':
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        main()

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
