import numpy as np
import csv
import pickle

def load_data(csv_file_name, useless_rows, survived_id = False):
    bias = 1 if survived_id == True else 0
    f = open(csv_file_name)
    reader = csv.reader(f)
    origin_data = list(reader)
    del (origin_data[0])
    # origin_data = list(map(normalize_data, origin_data))
    origin_data = [normalize_data(row, bias) for row in origin_data]

    origin_data = np.array(origin_data)
    examples_num = origin_data.shape[0]
    # result = np.loadtxt(csv_file_name, delimiter=",")
    X = np.transpose(np.delete(origin_data, useless_rows, axis=1)).astype(float)
    Y = []
    if survived_id:
        Y = origin_data[:, survived_id].reshape((1, examples_num)).astype(float)
    return X, Y


def normalize_data(row, bias):
    row[3 + bias] = 1 if row[3 + bias] == 'male' else 0
    if len(row[10 + bias]) == 1:
        row[10 + bias] = ord(row[10 + bias]) - 64
    return [0 if x == '' else x for x in row]


def map_sex_to_number(X_temp):
    X_temp[1, :] = X_temp[1, :] == 'male'

def calculate_precision_rate(Y_predict, Y):
    num_examples = Y.shape[1]
    Y_error = np.logical_xor(Y_predict, Y)
    precision_rate = 1.0 - np.sum(Y_error) / num_examples
    return precision_rate

def save_to_csv(Y, csv_file_name):
    index = np.array(range(892, 1310)).reshape(1, 418)
    result = np.concatenate((index, Y), axis=0)
    result = np.transpose(result)
    np.savetxt(csv_file_name, result, fmt='%i', header='PassengerId,Survived', comments='', delimiter=',')

def save_dict_to_pkl(parameter, pkl_file_name):
    f = open(pkl_file_name, 'wb')
    pickle.dump(parameter, f)
    f.close()

def load_dict_from_pkl(pkl_file_name):
    f = open(pkl_file_name, 'rb')
    parameters = pickle.load(f)
    return parameters