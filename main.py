import custom_network as cn
import numpy as np
import random

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

def iris_train():
    iris_x, iris_y = load_iris_dataset()

    # print(iris[0])
    # print(iris_x[0])
    # print(min_val, max_val, range)


    neural_network = cn.Network(epochs=10, learning_rate=0.0001)
    neural_network.load_model("good.nn")
    # neural_network.add_input_layer(inputs=4)
    # neural_network.add_hidden_layer(10, cn.sigmoid)
    # neural_network.add_hidden_layer(6, cn.sigmoid)
    # # neural_network.add_hidden_layer(3, cn.sigmoid)
    # neural_network.add_hidden_layer(3, cn.sigmoid)

    try:
        for i in range(3000000000000):
            for j in range(len(iris_x)):
                neural_network.train(iris_x[j], iris_y[j])
    except Exception as e:
        print(e)
        input()
        for i in range(1):
            for j in range(len(iris_x)):
                neural_network.train(iris_x[j], iris_y[j])


def iris_test():
    iris_x, iris_y = load_iris_dataset()
    neural_network = cn.Network(epochs=10, learning_rate=0.0001)
    neural_network.load_model("save.nn")
    test_index = random.randint(0, len(iris_x))

    hit = 0
    total = 0
    for i in range(len(iris_x)):
        test_x = iris_x[i]
        test_y = iris_y[i]
        predict = neural_network.predict(test_x)
        #print(predict, test_y)
        if predict == test_y:
            hit += 1
        total += 1

    print("Correct:", hit)
    print("Total:", total)
    print("Accuracy:", hit/total)
    # print("True:   ", test_y)
    # print("Predict:", predict)


if __name__ == '__main__':
    #iris_train()
    iris_test()
