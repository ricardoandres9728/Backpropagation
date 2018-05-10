from neuron import HiddenNeuron, OutputNeuron
import numpy as np


class Model(object):

    def __init__(self):
        self.hidden = [HiddenNeuron(40) for i in range(40)]
        self.output = OutputNeuron(40)

    def predict(self, input):
        temp = []
        for x in range(2):
            self.hidden[x].forward(input)
            temp.append(self.hidden[x].out)
        self.output.forward(temp)
        return self.output.out

    def train(self, inputs, targets):
        it = 0
        i = 0
        size = len(inputs)
        while it < 50000:
            if i == size:
                i = 0
            feature = inputs[i]
            print('\n\nFeature : ' + str(feature) + '\n')
            print('Output weights : ' + str(self.output.weights))
            print('Hidden 1 weights : ' + str(self.hidden[0].weights))
            print('Hidden 2 weights : ' + str(self.hidden[1].weights))
            temp = []
            for x in range(2):
                self.hidden[x].forward(feature)
                temp.append(self.hidden[x].out)
            self.output.forward(temp)
            self.output.backward(targets[i])
            deltas = [self.output.error]
            weights = [[self.output.weights[0]], [self.output.weights[1]]]
            for x in range(2):
                self.hidden[x].backward(deltas, weights[x])
            for x in range(2):
                self.hidden[x].update(feature)
            self.output.update(temp)
            it += 1
            i += 1
