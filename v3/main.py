import keras as k
import pandas as pd
import numpy as np
from v3.tool import data_preparation
from v3.deep_learning import training_parameter

x, y = data_preparation('../data/train.csv')
y = y.reshape(-1, 1)
x_test, _ = data_preparation('../data/test.csv')

x_train, x_dev = np.split(x, [800])
y_train, y_dev = np.split(y, [800])

predict = training_parameter(x_train, y_train, x_dev, y_dev, x_test)

predict = np.where(predict >= 0.5, 1, 0)
predict = pd.DataFrame(data=predict, index=np.arange(892, 1310))
predict.columns = ['Survived']
predict.to_csv('../data/submission.csv', index_label='PassengerId')

print('end')