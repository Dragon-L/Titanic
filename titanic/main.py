import csv
import numpy as np

USELESS_ROWS = [0, 1, 3, 8, 10]
SURVIVED_ID = 1
LAYER_DIM = [7, 5, 5, 3, 1]


def initialize_parameter(X, layer_dim):
    parameter = {};
    for l in range(1, len(layer_dim)):
        parameter['W' + str(l)] = np.random.rand(layer_dim[l], layer_dim[l - 1]) * 0.01
        parameter['b' + str(l)] = np.zeros((layer_dim[l], 1))
    return parameter


def relu(x):
    return np.maximum(x, 0)


def sigmod(x):
    return 1 / (1 + np.exp(-x))


def forward_propagation(X, parameter, layers_num):
    cache = {}
    A = X
    for l in range(1, layers_num - 1):
        np.dot(parameter['W' + str(l)], A)
        cache['Z' + str(l)] = np.dot(parameter['W' + str(l)], A) + parameter['b' + str(l)]
        cache['A' + str(l)] = relu(cache['Z' + str(l)])
        A = cache['A' + str(l)]
    cache['Z' + str(layers_num - 1)] = np.dot(parameter['W' + str(layers_num - 1)], A) + parameter['b' + str(layers_num - 1)]
    Y = sigmod(cache['Z' + str(layers_num - 1)])
    return Y, cache


def load_data(csv_file_name, useless_rows, survived_id):
    f = open(csv_file_name)
    reader = csv.reader(f)
    origin_data = list(reader)
    del (origin_data[0])
    origin_data = list(map(normalize_data, origin_data))

    origin_data = np.array(origin_data)
    examples_num = origin_data.shape[0]
    # result = np.loadtxt(csv_file_name, delimiter=",")
    X = np.transpose(np.delete(origin_data, useless_rows, axis=1)).astype(float)
    Y = origin_data[:, survived_id].reshape((1, examples_num)).astype(float)
    return X, Y


def normalize_data(row):
    row[4] = 1 if row[4] == 'male' else 0
    if len(row[11]) == 1:
        row[11] = ord(row[11]) - 64
    return [0 if x == '' else x for x in row]


def map_sex_to_number(X_temp):
    X_temp[1, :] = X_temp[1, :] == 'male'


def caculate_cost(Y, Y_train):
    loss = -(np.multiply(Y_train, np.log(Y)) + np.multiply(1 - Y_train, np.log(1 - Y)))
    cost = np.sum(loss) / Y.shape[1]
    return cost

def main():
    X_train, Y_train = load_data("../data/train.csv", USELESS_ROWS, SURVIVED_ID)
    parameter = initialize_parameter(X_train, LAYER_DIM)
    Y, cache = forward_propagation(X_train, parameter, len(LAYER_DIM))
    cost = caculate_cost(Y, Y_train)
    print(cost)


main()
