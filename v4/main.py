import tensorflow as tf
import numpy as np
import pandas as pd


def get_first_letter_and_set_N_when_value_is_nan(df):
    return pd.Series(['N' if pd.isnull(value) else value[0] for value in df['Cabin']], name='Cabin')


def fill_missing_age_with_average(df):
    df['Age'] = df.Age.fillna(df.Age.mean())
    return df


def fill_missing_fare_with_average(df):
    df['Fare'] = df.Fare.fillna(df.Fare.mean())
    return df


def data_preparation(input_file, train_file, dev_file):
    df = pd.read_csv(input_file)

    del df['PassengerId']
    del df['Name']
    del df['Ticket']

    df = fill_missing_age_with_average(df)
    df = fill_missing_fare_with_average(df)

    df['Cabin'] = get_first_letter_and_set_N_when_value_is_nan(df)
    train_df = df.iloc[:800, :]
    dev_df = df.iloc[800:, :]
    train_df.to_csv(train_file, index=False, header=False)
    dev_df.to_csv(dev_file, index=False, header=False)
    # print(df.describe())
    # print(df.info())


def decode_line(row):
    print(row)
    cols = tf.decode_csv(row, record_defaults=[[0], [3], ['male'], [22.0], [1], [0], [7.25], ['N'], ['S']])
    features = {
        'Pclass': cols[1],
        'Sex': cols[2],
        'Age': cols[3],
        'SibSp': cols[4],
        'Parch': cols[5],
        'Fare': cols[6],
        'Cabin': cols[7],
        'Embarked': cols[8]
    }
    label = cols[0]
    return features, label


def train_input_fn():
    features, label = train_dataset.make_one_shot_iterator().get_next()
    return features, label


def eval_input_fn():
    features, label = dev_dataset.make_one_shot_iterator().get_next()
    return features, label


# def train_input_fun(x):
#     dataset = tf.data.Dataset.from_tensor_slices(x)
#     features = dataset.make_one_shot_iterator().get_next()


train_set = '../data/handled_train.csv'
dev_set = '../data/handled_dev.csv'
data_preparation('../data/train.csv', train_set, dev_set)

featcols = [
    tf.feature_column.numeric_column('Pclass'),
    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('Sex', ['male', 'female'])),
    tf.feature_column.numeric_column('Age', default_value=-1.0),
    tf.feature_column.numeric_column('SibSp'),
    tf.feature_column.numeric_column('Parch'),
    tf.feature_column.numeric_column('Fare', default_value=-1.0),
    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('Cabin', ['N', 'C', 'B', 'D', 'E', 'A', 'F', 'G', 'T'])),
    tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list('Embarked', ['S', 'C', 'Q']))
]

estimator = tf.estimator.DNNRegressor(hidden_units=[512, 256, 256, 128, 128, 64, 64, 32, 32], feature_columns=featcols, model_dir='../output/')
train_dataset = tf.data.TextLineDataset(train_set).map(decode_line).batch(128)
dev_dataset = tf.data.TextLineDataset(dev_set).map(decode_line).batch(32)
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=200)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=20)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
estimator.predict()





