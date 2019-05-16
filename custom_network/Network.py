import numpy as np
import random
import pickle


class Layer:
    def __init__(self, neurons, activation_function, previous_neurons):
        self.neurons = neurons
        self.previous_neurons = previous_neurons
        self.activation_function = activation_function

        self.weights = []
        self.old_weights = []
        for i in range(self.neurons):
            row = []
            for j in range(self.previous_neurons):
                row.append(random.random())
            self.weights.append(row)

        self.weights = np.array(self.weights)
        self.offset_neurons = [random.random()] * self.neurons

        self.output = None
        self.before_activation = None
        self.sigm = [0] * self.neurons

    def fit(self, data_vector):

        data_vector = np.array(list(map(np.sum, data_vector * self.weights)))
        data_vector += self.offset_neurons
        self.before_activation = data_vector
        data_vector = np.array(list(map(self.activation_function, data_vector)))
        # for index, value in enumerate(data_vector):
        #     data_vector[index] = self.activation_function(value)
        self.output = data_vector
        self.old_weights = self.weights
        return data_vector


class OutputLayer:
    def __init__(self, neurons, previous_neurons):
        self.previous_neurons = previous_neurons
        self.neurons = neurons
        self.weights = np.array([[0.5] * self.previous_neurons] * self.neurons)
        self.offset_neurons = [0.5] * self.neurons
        self.before_activation = None

    def fit(self, data_vector):
        data_vector = np.array(list(map(np.sum, data_vector * self.weights)))
        data_vector += self.offset_neurons
        self.before_activation = data_vector
        return data_vector


class InputLayer:
    def __init__(self, inputs):
        self.inputs = inputs

        self.vector = None

    def fit(self, inputs: list):
        if len(inputs) != self.inputs:
            raise Exception("Размер входного вектора,"
                            " не соответсует объявленному.")

        self.vector = np.array(inputs)

    def get_data(self):
        return self.vector


class Network:
    def __init__(self, epochs: int, learning_rate: float, target=0.1):
        self.input_layer = None

        self.hidden_layers = []
        self.target = target
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.accuracy = 0
        self.count = 0
        self.iterations = 0

    def predict(self, train_data_x):
        # forward
        self.input_layer.fit(inputs=train_data_x)
        data_vector = self.input_layer.vector

        for layer in self.hidden_layers:
            data_vector = layer.fit(data_vector)

        max = 0
        index = 0
        for i, v in enumerate(data_vector):
            if v > max:
                max = v
                index = i
        predict = [0, 0, 0]
        predict[index] = 1
        return predict

    def train(self, dataset_x, dataset_y):
        for epoch in range(self.epochs):
            index_list = list(range(len(dataset_x)))
            random.shuffle(index_list)
            for pair_index in index_list:
                train_data_x = dataset_x[pair_index]
                train_data_y = dataset_y[pair_index]
                # forward
                self.input_layer.fit(inputs=train_data_x)
                data_vector = self.input_layer.vector

                for layer in self.hidden_layers:
                    data_vector = layer.fit(data_vector)

                # `print("output", data_vector)

                output_diff = data_vector - train_data_y
                mse = np.array(list(map(lambda x: x ** 2, output_diff)))
                max = 0
                index = 0
                for i, v in enumerate(data_vector):
                    if v > max:
                        max = v
                        index = i
                predict = [0, 0, 0]
                predict[index] = 1
                if predict == train_data_y:
                    self.accuracy += 1
                self.count += 1
                self.iterations += 1
                if pair_index == index_list[-1]:
                    loss = np.sum(mse)
                    print("Loss:", abs(np.sum(mse)), "Accuracy:", self.accuracy / self.count)
                    self.save_model("save.nn")

                    if loss <= self.target:
                        print(f"Target was reached. Iterations: {self.iterations},"
                              f" epochs: {epoch}")
                        return

                    self.count = 0
                    self.accuracy = 0

                # back
                # print(data_vector)
                # Цикл, проходящий по слоям сети в обратном порядке
                for index, layer in enumerate(reversed(self.hidden_layers)):
                    # Индекс слоя относительно self.hidden_layers
                    true_index = len(self.hidden_layers) - index - 1
                    # print(index)
                    if index != 0:
                        prev_layer = self.hidden_layers[true_index + 1]
                        output_diff = layer.neurons * [0]
                        for neuron_index in range(layer.neurons):
                            for prev_neuron_index in range(prev_layer.neurons):
                                output_diff[neuron_index] += prev_layer.sigm[prev_neuron_index] * \
                                                             prev_layer.old_weights[prev_neuron_index][neuron_index]

                    # Цикл, проходящий по нейронам в выбранном слое
                    for neuron_index in range(layer.neurons):
                        sigm = 2 * output_diff[neuron_index] * sigmoid_derivative(layer.before_activation[neuron_index])
                        layer.sigm[neuron_index] = sigm
                        dB = sigm * self.learning_rate
                        layer.offset_neurons[neuron_index] -= dB
                        # Цикл проходящий по связанным нейронам с предыдущего слоя
                        for previous_neuron_index, previous_neuron in enumerate(
                                self.hidden_layers[true_index - 1].output):
                            # Изменение весов
                            dW = self.learning_rate * sigm * previous_neuron
                            layer.weights[neuron_index][previous_neuron_index] -= dW
                            # Изменение весов нейрона смешения
        print("All epoch was ended.")

    def save_model(self, name):
        dump = {}
        dump["input"] = self.input_layer
        dump["hidden"] = self.hidden_layers
        with open(name, 'wb') as f:
            pickle.dump(dump, f)
        print("Saved")

    def load_model(self, path):
        with open(path, 'rb') as f:
            layers = pickle.load(f)
            self.input_layer = layers["input"]
            self.hidden_layers = layers["hidden"]

    def add_input_layer(self, inputs):
        input_layer = InputLayer(inputs)
        self.input_layer = input_layer

    def add_hidden_layer(self, neurons, activation_function):
        if len(self.hidden_layers) == 0:
            previous_neurons = self.input_layer.inputs
        else:
            previous_neurons = self.hidden_layers[-1].neurons
        new_layer = Layer(neurons, activation_function, previous_neurons)

        self.hidden_layers.append(new_layer)

    def add_output_layer(self, neurons):
        previous_neurons = self.hidden_layers[-1].neurons
        output_layer = OutputLayer(neurons, previous_neurons)
        self.output_layer = output_layer


def sigmoid_derivative(value):
    return sigmoid(value) * (1 - sigmoid(value))


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def linear(value):
    if value < 0:
        return 0
    elif value > 1:
        return 1
    return value
