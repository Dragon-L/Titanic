import csv
import numpy as np

USELESS_ROWS = [0, 1, 3, 8, 10]
SURVIVED_ID = 1
LAYER_DIM = [5, 5, 3, 1]


# def initialize_parameter(X, layer_dim):
    # layer0 = X.shape[0]
    # layer_dim = layer0 + layer_dim

    # print(layer_dim)
    # parameter = {};
    # for l in range(len(layer_dim)):
    #     parameter['W'+str(l+1)] = np.random.rand(layer_dim[l], )


def main():
    X_train, Y_train = load_data("../data/train.csv", USELESS_ROWS, SURVIVED_ID)
    print(X_train.shape)
    print(Y_train.shape)
    # initialize_parameter(X_train, LAYER_DIM)


def load_data(csv_file_name, useless_rows, survived_id):
    f = open(csv_file_name)
    reader = csv.reader(f)
    origin_data = list(reader)
    del (origin_data[0])
    origin_data = list(map(normalize_data, origin_data))

    origin_data = np.array(origin_data)
    examples_num = origin_data.shape[0]
    # result = np.loadtxt(csv_file_name, delimiter=",")
    X = np.transpose(np.delete(origin_data, useless_rows, axis=1))
    Y = origin_data[:, survived_id].reshape((1, examples_num))
    return X, Y


def normalize_data(row):
    row[4] = 1 if row[4] == 'male' else 0
    if len(row[11]) == 1:
        row[11] = ord(row[11]) - 64
    return row


def map_sex_to_number(X_temp):
    X_temp[1, :] = X_temp[1, :] == 'male'


main()
