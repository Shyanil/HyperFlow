from FlowUnit_module import FlowUnit
import random
import warnings


class Nuron:
    def __init__(self, number_of_neurons):
        self.w = [random.uniform(-1, 1) for _ in range(number_of_neurons)]
        self.b = random.uniform(-1, 1)

    def __call__(self, X, function=None):
        out = sum((wi * xi for wi, xi in zip(self.w, X)), self.b)
        out = FlowUnit(out)
        if function == None:
            return out
        if function == "sigmoid":
            return out.sigmoid()
        elif function == "tanh":
            return out.tanh()
        elif function == "relu":
            return out.relu()
        elif function == "leaky_relu":
            return out.leaky_relu()
        else:
            raise ValueError(f"Unsupported activation function: {function}")

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, number_of_in_neurons, number_of_out_neurons):
        self.neurons = [
            Nuron(number_of_in_neurons) for _ in range(number_of_out_neurons)
        ]

    def __call__(self, X):
        outs = [n(X) for n in self.neurons]
        if len(outs) == 1:
            return outs[0]
        else:
            return outs

    def parameters(self):
        params = []
        for nuron in self.neurons:
            p = nuron.parameters()
            params.append(p)
        return params


class NeuralNetwork:
    def __init__(self, input_size, layers_sizes):
        self.layers = []
        prev_size = input_size
        for size in layers_sizes:
            self.layers.append(Layer(prev_size, size))
            prev_size = size

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params
