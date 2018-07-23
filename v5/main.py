from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import KFold
import xgboost
import numpy as np
import pandas as pd


def get_base_prediction(model, x, y, test):
    kf = KFold(n_splits=5)

    prediction_train = np.zeros(y.shape)
    prediction_test = np.zeros((5, test.shape[0]))
    for i, (train_index, test_index) in enumerate(kf.split(x)):
        subset_train_x = x[train_index]
        subset_train_y = y[train_index]
        subset_test_x = x[test_index]

        model.fit(subset_train_x, subset_train_y)

        prediction_train[test_index] = rfc.predict(subset_test_x)
        prediction_test[i, :] = rfc.predict(test)

    prediction_test = prediction_test.mean(axis=0)
    return prediction_train.reshape((-1, 1)), prediction_test.reshape((-1, 1))


def print_accurate_rate(predictions, labels):
    predictions = predictions.reshape(-1, )
    print(np.divide(np.sum(predictions == labels), labels.shape[0]))


base_train_x = np.genfromtxt('../data/train_x.csv', delimiter=',')
base_train_y = np.genfromtxt('../data/train_y.csv', delimiter=',')
base_test_x = np.genfromtxt('../data/test_x.csv', delimiter=',')

rfc = RandomForestClassifier(n_jobs=-1, n_estimators=500, max_depth=6, max_features='sqrt')
adc = AdaBoostClassifier(n_estimators=500, learning_rate=0.75)
gbc = GradientBoostingClassifier(n_estimators=500, max_depth=5)
etc = ExtraTreesClassifier(n_jobs=-1, n_estimators=500, max_depth=8)

rfc_train_x, rfc_test_x = get_base_prediction(rfc, base_train_x, base_train_y, base_test_x)
print_accurate_rate(rfc_train_x, base_train_y)
adc_train_x, adc_test_x = get_base_prediction(adc, base_train_x, base_train_y, base_test_x)
print_accurate_rate(adc_train_x, base_train_y)
gbc_train_x, gbc_test_x = get_base_prediction(gbc, base_train_x, base_train_y, base_test_x)
print_accurate_rate(gbc_train_x, base_train_y)
etc_train_x, etc_test_x = get_base_prediction(etc, base_train_x, base_train_y, base_test_x)
print_accurate_rate(etc_train_x, base_train_y)

second_train_x = np.concatenate((rfc_train_x, adc_train_x, gbc_train_x,  etc_train_x), axis=1)
second_test_x = np.concatenate((rfc_test_x, adc_test_x, gbc_test_x, etc_test_x), axis=1)

xgb = xgboost.XGBClassifier(n_estimators=2000, max_depth=4)
xgb.fit(second_train_x, base_train_y)
predictions = xgb.predict(second_test_x).astype(int)
submission = pd.DataFrame(predictions, index=np.arange(892, 1310), columns=['Survived'])
submission.to_csv('../data/submission.csv', index_label='PassengerId')












