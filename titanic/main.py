import csv
import numpy as np

USELESS_ROWS = [0, 1, 3, 8, 10]
SURVIVED_ID = 1
LAYER_DIM = [7, 5, 5, 3, 1]


def initialize_parameters(X, layer_dim):
    parameter = {};
    for l in range(1, len(layer_dim)):
        parameter['W' + str(l)] = np.random.randn(layer_dim[l], layer_dim[l - 1]) * 0.01
        parameter['b' + str(l)] = np.zeros((layer_dim[l], 1))
    return parameter


def relu(x):
    return np.maximum(x, 0)


def sigmod(x):
    return 1 / (1 + np.exp(-x))


def forward_propagation(X, parameter, layers_len):
    cache = {}
    A_prev = X
    cache['A0'] = X
    for l in range(1, layers_len):
        np.dot(parameter['W' + str(l)], A_prev)
        cache['Z' + str(l)] = np.dot(parameter['W' + str(l)], A_prev) + parameter['b' + str(l)]
        cache['A' + str(l)] = relu(cache['Z' + str(l)])
        A_prev = cache['A' + str(l)]
    cache['Z' + str(layers_len)] = np.dot(parameter['W' + str(layers_len)], A_prev) + parameter['b' + str(layers_len)]
    Y = sigmod(cache['Z' + str(layers_len)])
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


def relu_derivative(X):
    return X >= 0


def backward_propagation(Y_train, Y, cache, parameter, layer_len):
    grads = {}
    m = Y.shape[1]
    grads['dZ' + str(layer_len)] = Y - Y_train
    grads['dW' + str(layer_len)] = 1 / m * np.dot(grads['dZ' + str(layer_len)], cache['A' + str(layer_len - 1)].T)
    grads['db' + str(layer_len)] = 1 / m * np.sum(grads['dZ' + str(layer_len)], axis=1, keepdims=True)

    for l in reversed(range(1, layer_len)):
        grads['dZ' + str(l)] = np.multiply(np.dot(parameter['W' + str(l + 1)].T, grads['dZ' + str(l + 1)]),
                                           relu_derivative(cache['Z' + str(l)]))
        grads['dW' + str(l)] = 1 / m * np.dot(grads['dZ' + str(l)], cache['A' + str(l - 1)].T)
        grads['db' + str(l)] = 1 / m * np.sum(grads['dZ' + str(l)], axis=1, keepdims=True)
    return grads


def update_parameters(parameters, grads, layers_len, learning_rate):
    for l in range(1, layers_len + 1):
        parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]


def dictionary_to_vector(parameters, layer_len, key_W, key_b):
    W1 = parameters[key_W + '1']
    b1 = parameters[key_b + '1']
    vector = W1.reshape(1, W1.shape[0] * W1.shape[1])
    vector = np.append(vector, b1.reshape(1, b1.shape[0]))
    for l in range(2, layer_len + 1):
        Wl = parameters[key_W + str(l)]
        bl = parameters[key_b + str(l)]
        vector = np.append(vector, Wl.reshape(1, Wl.shape[0] * Wl.shape[1]))
        vector = np.append(vector, bl.reshape(1, bl.shape[0]))
    return vector


def vector_to_parameters(vector, layer_dim):
    index_start = 0
    index_end = 0
    parameters = {}
    for l in range(1, len(layer_dim)):
        index_end = index_start + layer_dim[l] * layer_dim[l - 1]
        parameters['W' + str(l)] = vector[index_start:index_end].reshape((layer_dim[l], layer_dim[l-1]))
        index_start = index_end
        index_end = index_start + layer_dim[l]
        parameters['b' + str(l)] = vector[index_start:index_end].reshape((layer_dim[l], 1))
        index_start = index_end
    return parameters

def check_gradient(X_train, Y_train, grads, parameters, layer_dim, epsilon=1e-7):
    grads_approx = []
    theta = dictionary_to_vector(parameters, len(layer_dim) - 1, 'W', 'b')
    parameters_num = len(theta)
    for i in range(parameters_num):
        theta_plus = np.copy(theta)
        theta_plus[i] += epsilon
        theta_min = np.copy(theta)
        theta_min[i] -= epsilon
        Y_plus, _ = forward_propagation(X_train, vector_to_parameters(theta_plus, layer_dim), len(layer_dim) - 1)
        Y_min, _ = forward_propagation(X_train, vector_to_parameters(theta_min, layer_dim), len(layer_dim) - 1)
        cost_plus = caculate_cost(Y_plus ,Y_train)
        cost_min = caculate_cost(Y_min ,Y_train)
        gradient = (cost_plus - cost_min) / (2 * epsilon)
        grads_approx.append(gradient)

    grads_vector = dictionary_to_vector(grads, len(layer_dim) - 1, 'dW', 'db')
    grads_error = np.linalg.norm(grads_approx - grads_vector) / (np.linalg.norm(grads_vector) + np.linalg.norm(grads_approx))
    return grads_error

    # parameters_changed = vector_to_parameters(theta, layer_dim)


def four_layer_model(learning_rate=0.0075, num_iterations=1000):
    X_train, Y_train = load_data("../data/train.csv", USELESS_ROWS, SURVIVED_ID)
    parameters = initialize_parameters(X_train, LAYER_DIM)
    layers_len = len(LAYER_DIM) - 1

    for i in range(num_iterations):
        Y, cache = forward_propagation(X_train, parameters, layers_len)
        cost = caculate_cost(Y, Y_train)
        grads = backward_propagation(Y_train, Y, cache, parameters, layers_len)

        # grads_error = check_gradient(X_train, Y_train, grads, parameters, LAYER_DIM)

        update_parameters(parameters, grads, layers_len, learning_rate)

        if  i%100 == 0:
            print("Cost after iterations %i: %f" %(i, cost))
            # print("Grads error is:" + str(grads_error))
            # print('gradient is %f' % (grads['dW1'][0,0]))
            # print(grads['db1'])
            # print(parameters['W1'][0,0])

    return parameters


four_layer_model()
