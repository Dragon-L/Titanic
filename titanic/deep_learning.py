import csv
import numpy as np
from titanic.tool import load_dict_from_pkl


def initialize_parameters(X, layer_dim):
    parameter = {};
    for l in range(1, len(layer_dim)):
        small = np.power(np.divide(1, layer_dim[l - 1]), 0.5)
        parameter['W' + str(l)] = np.random.randn(layer_dim[l], layer_dim[l - 1]) * small
        parameter['b' + str(l)] = np.zeros((layer_dim[l], 1))
    return parameter


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward_propagation(X, parameter, layers_len, activation):
    cache = {}
    A_prev = X
    cache['A0'] = X
    for l in range(1, layers_len):
        np.dot(parameter['W' + str(l)], A_prev)
        cache['Z' + str(l)] = np.dot(parameter['W' + str(l)], A_prev) + parameter['b' + str(l)]
        if activation == 'relu':
            cache['A' + str(l)] = relu(cache['Z' + str(l)])
        elif activation == 'sigmoid':
            cache['A' + str(l)] = sigmoid(cache['Z' + str(l)])
        A_prev = cache['A' + str(l)]
    cache['Z' + str(layers_len)] = np.dot(parameter['W' + str(layers_len)], A_prev) + parameter['b' + str(layers_len)]
    Y = sigmoid(cache['Z' + str(layers_len)])
    return Y, cache


def calculate_cost(Y, Y_train):
    loss = -(np.multiply(Y_train, np.log(Y)) + np.multiply(1 - Y_train, np.log(1 - Y)))
    cost = np.sum(loss) / Y.shape[1]
    cost = np.squeeze(cost)
    return cost


def relu_derivative(X):
    return X >= 0


def sigmoid_derivative(X):
    return sigmoid(X) * (1 - sigmoid(X))


def backward_propagation(Y_train, Y, cache, parameter, layer_len, activation):
    grads = {}
    m = Y.shape[1]
    grads['dZ' + str(layer_len)] = Y - Y_train
    grads['dW' + str(layer_len)] = 1 / m * np.dot(grads['dZ' + str(layer_len)], cache['A' + str(layer_len - 1)].T)
    grads['db' + str(layer_len)] = 1 / m * np.sum(grads['dZ' + str(layer_len)], axis=1, keepdims=True)

    for l in reversed(range(1, layer_len)):
        if activation == 'relu':
            grads['dZ' + str(l)] = np.multiply(np.dot(parameter['W' + str(l + 1)].T, grads['dZ' + str(l + 1)]),
                                               relu_derivative(cache['Z' + str(l)]))
        elif activation == 'sigmoid':
            grads['dZ' + str(l)] = np.multiply(np.dot(parameter['W' + str(l + 1)].T, grads['dZ' + str(l + 1)]),
                                               sigmoid_derivative(cache['Z' + str(l)]))
        grads['dW' + str(l)] = 1 / m * np.dot(grads['dZ' + str(l)], cache['A' + str(l - 1)].T)
        grads['db' + str(l)] = 1 / m * np.sum(grads['dZ' + str(l)], axis=1, keepdims=True)
    return grads


def update_parameters(parameters, grads, layers_len, learning_rate):
    for l in range(1, layers_len + 1):
        parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]

    return parameters


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
        parameters['W' + str(l)] = vector[index_start:index_end].reshape((layer_dim[l], layer_dim[l - 1]))
        index_start = index_end
        index_end = index_start + layer_dim[l]
        parameters['b' + str(l)] = vector[index_start:index_end].reshape((layer_dim[l], 1))
        index_start = index_end
    return parameters


def check_gradient(X_train, Y_train, grads, parameters, layer_dim, activation, epsilon=1e-7):
    grads_approx = []
    theta = dictionary_to_vector(parameters, len(layer_dim) - 1, 'W', 'b')
    parameters_num = len(theta)
    for i in range(parameters_num):
        theta_plus = np.copy(theta)
        theta_plus[i] += epsilon
        theta_min = np.copy(theta)
        theta_min[i] -= epsilon
        Y_plus, _ = forward_propagation(X_train, vector_to_parameters(theta_plus, layer_dim), len(layer_dim) - 1,
                                        activation)
        Y_min, _ = forward_propagation(X_train, vector_to_parameters(theta_min, layer_dim), len(layer_dim) - 1,
                                       activation)
        cost_plus = calculate_cost(Y_plus, Y_train)
        cost_min = calculate_cost(Y_min, Y_train)
        gradient = (cost_plus - cost_min) / (2 * epsilon)
        grads_approx.append(gradient)

    grads_vector = dictionary_to_vector(grads, len(layer_dim) - 1, 'dW', 'db')
    grads_error = np.linalg.norm(grads_approx - grads_vector) / (
                np.linalg.norm(grads_vector) + np.linalg.norm(grads_approx))
    return grads_error, grads_vector, grads_approx

    # parameters_changed = vector_to_parameters(theta, layer_dim)


def predict(X, parameters, layers_len, activation):
    Y, _ = forward_propagation(X, parameters, layers_len, activation)
    Y_predict = (Y > 0.5).astype(int)
    return Y_predict


def training_parameters(X_train, Y_train, layer_dim, activation, parameters, learning_rate=0.1, num_iterations=100000):
    layers_len = len(layer_dim) - 1

    prev_cost = 10000
    wrong_times = 0

    for i in range(num_iterations):
        Y, cache = forward_propagation(X_train, parameters, layers_len, activation)
        cost = calculate_cost(Y, Y_train)
        grads = backward_propagation(Y_train, Y, cache, parameters, layers_len, activation)

        # grads_error, grads_vector, grads_approx = check_gradient(X_train, Y_train, grads, parameters, LAYER_DIM, activation)

        parameters = update_parameters(parameters, grads, layers_len, learning_rate)

        if i % 1000 == 0:
            print("Cost after iterations %i: %f" % (i, cost))

            # print("gradient is %f" % (grads['dW2'][0, 0]))
            if cost > prev_cost:
                wrong_times += 1
            if wrong_times >= 5:
                wrong_times = 0
                learning_rate = learning_rate * 4 / 5
                print("====================Current learning rate is %f================" % learning_rate)

            prev_cost = cost

    return parameters


