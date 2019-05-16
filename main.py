import cv2
import os

import custom_network as cn
import numpy as np
import random
import pickle


def load_iris_dataset():
    iris = []
    with open("iris.txt", "r") as f:
        for i in f:
            iris.append(i.replace("\n", "").split(","))

    iris_x = []
    iris_y = []
    for index, row in enumerate(iris):
        iris_x.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
        if row[4] == "setosa":
            iris_y.append([1, 0, 0])
        elif row[4] == "versicolor":
            iris_y.append([0, 1, 0])
        else:
            iris_y.append([0, 0, 1])

    min_val = [100, 100, 100, 100]
    max_val = [0, 0, 0, 0]

    for index, row in enumerate(iris_x):
        for index_, value_ in enumerate(row):
            if value_ > max_val[index_]:
                max_val[index_] = value_
            if value_ < min_val[index_]:
                min_val[index_] = value_
    range_ = [0, 0, 0, 0]
    for index, value in enumerate(max_val):
        range_[index] = max_val[index] - min_val[index]

    for index, row in enumerate(iris_x):
        for index_, value_ in enumerate(row):
            iris_x[index][index_] = (value_ - min_val[index_]) / range_[index_]

    return iris_x, iris_y


def iris_train(data_x=None, data_y=None):
    if data_x is not None and data_y is not None:
        iris_x, iris_y = data_x, data_y
    else:
        iris_x, iris_y = load_iris_dataset()

    # print(iris[0])
    # print(iris_x[0])
    # print(min_val, max_val, range)

    targets = [0.9, 0.5, 0.1]
    learning_rates = [0.01, 0.0001]
    neurons = [[6, 3], [10, 3]]
    neural_network = cn.Network(epochs=1000, learning_rate=0.01, target=0.000001)
    print("Загрузить? (y/n)")
    answer = input()
    if answer == "y":
        neural_network.load_model("save.nn")
    else:
        neural_network.add_input_layer(inputs=4)
        neural_network.add_hidden_layer(6, cn.sigmoid)
        neural_network.add_hidden_layer(3, cn.sigmoid)
    neural_network.train(iris_x, iris_y)

    # for i in targets:
    #     for j in learning_rates:
    #         for k in neurons:
    #             neural_network = cn.Network(epochs=1000, learning_rate=j, target=i)
    #             neural_network.add_input_layer(inputs=4)
    #             for neuron in k:
    #                 neural_network.add_hidden_layer(neuron, cn.sigmoid)
    #             print(f"Target: {i}, Learning rate: {j}, Neurons: {k}")
    #             input()
    #             neural_network.train(iris_x, iris_y)
    #             input()


def iris_test(x, y):
    iris_x, iris_y = x, y
    neural_network = cn.Network(epochs=1, learning_rate=0.0001)
    neural_network.load_model("save.nn")
    test_index = random.randint(0, len(iris_x))

    hit = 0
    total = 0
    class_accuracy = [0, 0, 0]

    for i in range(len(iris_x)):
        test_x = iris_x[i]
        test_y = iris_y[i]
        predict = neural_network.predict(test_x)
        # print(predict, test_y)
        if predict == test_y:
            hit += 1
            if predict == [1, 0, 0]:
                class_accuracy[0] += 1
            elif predict == [0, 1, 0]:
                class_accuracy[1] += 1
            else:
                class_accuracy[2] += 1

        total += 1

    print("Correct:", hit)
    print("Total:", total)
    print("Accuracy:", hit / total)
    print("Class accuracy: ", list(map(lambda x: x / 10, class_accuracy)))


def iris(train=True):
    irisx, irisy = load_iris_dataset()
    random.seed(73)
    indexes = []
    offset = 50
    for k in range(3):
        for i in range(10):
            num = random.randint(k * offset, k * offset + offset)
            while num in indexes:
                num = random.randint(k * offset, k * offset + offset)
            indexes.append(num)
    indexes.sort()
    test_x = []
    test_y = []
    for i in indexes:
        test_x.append(irisx[i])
        test_y.append(irisy[i])

    for i in reversed(indexes):
        irisx.pop(i)
        irisy.pop(i)

    if train:
        iris_train(data_x=irisx, data_y=irisy)
    else:
        iris_test(test_x, test_y)


def mnist(train=True):
    mnist_data = read_mnist_data()
    net_train(mnist_data["train_x"], mnist_data["train_y"])


def net_train(data_x, data_y):
    neural_network = cn.Network(epochs=1000, learning_rate=0.0001, target=0.000001)
    print("Загрузить? (y/n)")
    answer = input()
    if answer == "y":
        neural_network.load_model("save.nn")
    else:
        neural_network.add_input_layer(inputs=784)
        neural_network.add_hidden_layer(6, cn.sigmoid)
        # neural_network.add_hidden_layer(16, cn.sigmoid)
        neural_network.add_hidden_layer(10, cn.sigmoid)
    neural_network.train(data_x, data_y)


def read_mnist_data():
    mnist_data = {"train": {}, "test": {}}

    root = "trainingSample/"
    for i in os.listdir(root):
        mnist_data["train"][i] = []
        for j in os.listdir(root + i):
            image = cv2.imread(root + i + "/" + j, cv2.IMREAD_GRAYSCALE)
            image = np.apply_along_axis(lambda x: x / 255, 0, image)
            mnist_data["train"][i].append(image)

    for key, value in mnist_data["train"].items():
        mnist_data["train"][key] = value[:50]
        mnist_data["test"][key] = value[50:]

    data = {"train_x": [], "train_y": [], "test_x": [], "test_y": []}

    for key, value in mnist_data["train"].items():
        y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        y[int(key)] = 1
        for i in value:
            row = []
            for k in i:
                for l in k:
                    row.append(l)
            data["train_x"].append(row)
            data["train_y"].append(y)

    for key, value in mnist_data["test"].items():
        y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        y[int(key)] = 1
        for i in value:
            row = []
            for k in i:
                for l in k:
                    row.append(l)
            data["test_x"].append(row)
            data["test_y"].append(y)

    return data


def image_normalization(image):
    pass


if __name__ == '__main__':
    iris(train=False)
    #mnist()
