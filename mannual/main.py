from utils.tool import load_data, calculate_precision_rate, save_to_csv, save_dict_to_pkl, load_dict_from_pkl
from mannual.deep_learning import training_parameters, predict

USELESS_ROWS_FOR_TRAIN = [0, 1, 3, 8, 10]
USELESS_ROWS_FOR_TEST = [0, 2, 7, 9]
SURVIVED_ID = 1
LAYER_DIM = [7, 5, 5, 3, 3, 1]
activation = 'relu'

X_train, Y_train = load_data('../data/train.csv', USELESS_ROWS_FOR_TRAIN, SURVIVED_ID)
# parameters = initialize_parameters(X_train, LAYER_DIM)
parameters = load_dict_from_pkl('../data/parameters.pkl')
parameters = training_parameters(X_train, Y_train, LAYER_DIM, activation, parameters, 0.01)
save_dict_to_pkl(parameters, '../data/parameters.pkl')

Y_predict = predict(X_train, parameters, len(LAYER_DIM) - 1, activation)
precision_rate = calculate_precision_rate(Y_predict, Y_train)
print('precision rate is: %f' %precision_rate)

X_test, _ = load_data('../data/test.csv', USELESS_ROWS_FOR_TEST)
Y_test_predict = predict(X_test, parameters, len(LAYER_DIM) - 1, activation)
save_to_csv(Y_test_predict, '../data/submission.csv')

