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


def data_preparation(input_file):
    df = pd.read_csv(input_file)

    del df['PassengerId']
    del df['Name']
    del df['Ticket']

    df = fill_missing_age_with_average(df)
    df = fill_missing_fare_with_average(df)

    df['Cabin'] = get_first_letter_and_set_N_when_value_is_nan(df)
    return df


def decode_train_line(row):
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


def decode_test_line(row):
    cols = tf.decode_csv(row, record_defaults=[[3], ['male'], [22.0], [1], [0], [7.25], ['N'], ['S']])
    features = {
        'Pclass': cols[0],
        'Sex': cols[1],
        'Age': cols[2],
        'SibSp': cols[3],
        'Parch': cols[4],
        'Fare': cols[5],
        'Cabin': cols[6],
        'Embarked': cols[7]
    }
    return features


def train_input_fn(dataset):
    features, label = dataset.make_one_shot_iterator().get_next()
    return features, label


def eval_input_fn(dataset):
    features, label = dataset.make_one_shot_iterator().get_next()
    return features, label


def test_input_fn(dataset):
    features = dataset.make_one_shot_iterator().get_next()
    return features


train_set = '../data/handled_train.csv'
dev_set = '../data/handled_dev.csv'
test_set = '../data/handled_test.csv'
submission_file = '../data/submission.csv'

# train_df = data_preparation('../data/train.csv')
# test_df = data_preparation('../data/test.csv')
#
# train_df.iloc[:800, :].to_csv(train_set, index=False, header=False)
# train_df.iloc[800:, :].to_csv(dev_set, index=False, header=False)
# test_df.to_csv(test_set, index=False, header=False)

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

estimator = tf.estimator.DNNRegressor(hidden_units=[512, 256, 256, 128, 128, 64, 64, 32, 32], feature_columns=featcols,
                                      model_dir='../output/')
train_dataset = tf.data.TextLineDataset(train_set).map(decode_train_line).shuffle(1000).batch(128)
dev_dataset = tf.data.TextLineDataset(dev_set).map(decode_train_line).shuffle(1000).batch(32)
test_dataset = tf.data.TextLineDataset(test_set).map(decode_test_line).batch(32)

tf.summary.scalar('test_var', 1)
train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(train_dataset), max_steps=2040)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn(dev_dataset), steps=20)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

predictions = estimator.predict(input_fn=lambda: test_input_fn(test_dataset))

result = np.array([])
for pred in predictions:
    result = np.append(result, pred['predictions'])

result = np.where(result < 0.5, 0, 1)
submission = pd.DataFrame(data=result, index=np.arange(892, 1310), columns=['Survived'])
submission.to_csv(submission_file, index_label='PassengerId')

