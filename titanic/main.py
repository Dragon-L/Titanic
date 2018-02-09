import csv
import numpy as np

useless_rows = [0, 1, 3, 8, 10]
survived_id = 1

def load_data(csv_file_name):
    f = open(csv_file_name)
    reader = csv.reader(f)
    x = list(reader)
    del(x[0])
    print(x[1])
    x = list(map(normalize_data, x))
    print(x[1])


    # result = np.array(x)
    # result = np.loadtxt(csv_file_name, delimiter=",")

    # title = result[0,:]
    # title = title.reshape(1, title.shape[0])
    # title = np.delete(title, useless_rows, axis = 1)
    # print(title[0])
    #
    # result = np.transpose(np.delete(result, 0, axis = 0))
    # example_num = result.shape[1]
    # X = np.delete(result, useless_rows, axis = 0)
    # Y = result[survived_id, :].reshape((1, example_num))
    # return X, Y

def main():
    load_data("../data/train.csv")
    # X_train_temp, Y_train_temp = load_data("../data/train.csv")
    # X_train, Y_train = normalize_data(X_train_temp, Y_train_temp)
    # print(X_train[:, 0])
    # print(X_train[6, :])

def normalize_data(row):
    row[4] = 1 if row[4] == 'male' else 0
    if len(row[11]) == 1:
        row[11] = ord(row[11]) - 64
    # else:
    #     row[11] = 0
    return row
    # return row[11]
    # if row == 'male':
        # return 1


    # map_sex_to_number(X_temp)
    # map_embarked_to_number(X_temp)
    # return X_temp, Y_temp

def map_sex_to_number(X_temp):
    X_temp[1, :] = X_temp[1, :] == 'male'




main()
