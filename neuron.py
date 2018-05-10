import numpy as np
from random import uniform


class Neuron(object):

    def activation(self, fx):
        return 1 / (1 + np.exp(-fx))

    def __init__(self, dim, lrate):
        self.dim = dim
        self.weights = np.empty([dim])
        self.weights = [uniform(0, 1) for x in range(dim)]
        self.bias = uniform(0, 1)
        self.lrate = lrate
        self.out = None
        self.error = None

    def update(self, input):
        j = 0
        for i in input:
            delta = self.lrate * self.error
            self.weights[j] -= (delta * i)
            self.bias -= delta
            j += 1

    def forward(self, input):
        j = 0
        sum = self.bias
        for f in input:
            sum += f * self.weights[j]
            j += 1
        self.out = self.activation(sum)

    def backward(self):
        pass


class OutputNeuron(Neuron):

    def __init__(self, dim, lrate=0.5):
        super(OutputNeuron, self).__init__(dim, lrate)

    def backward(self, target):
        self.error = self.out * (1 - self.out) * (self.out - target)


class HiddenNeuron(Neuron):

    def __init__(self, dim, lrate=0.5):
        super(HiddenNeuron, self).__init__(dim, lrate)

    def backward(self, deltas, weights):
        sum = 0
        size = len(deltas)
        for x in range(size):
            sum += deltas[x] * weights[x]
        self.error = self.out * (1 - self.out) * sum
