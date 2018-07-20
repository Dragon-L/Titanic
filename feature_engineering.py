import pandas as pd


def get_first_letter(word):
    return word[0]


def fill_null_value(value):
    return 'N' if pd.isnull(value) else value


def process_data(df, drop_columns):
    for column in drop_columns:
        del df[column]

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Age'] = df['Age'].fillna(df['Age'].mean()).astype(int)
    df['Cabin'] = df['Cabin'].map(fill_null_value).map(get_first_letter).map({'N': 0, 'C': 1, 'B': 2, 'D': 3,'E': 4,'A':5,'F':6,'G':7,'T':8})
    df['Embarked'] = df['Embarked'].map(fill_null_value).map({'S': 0, 'C': 1, 'Q': 2, 'N': 3})

    return df


DROP_COLUMNS = ['PassengerId', 'Name', 'Ticket']

original_train_data = pd.read_csv('./data/train.csv')
original_test_data = pd.read_csv('./data/test.csv')

train_data = process_data(original_train_data, DROP_COLUMNS)
test_data = process_data(original_test_data, DROP_COLUMNS)
train_y = train_data['Survived']
del train_data['Survived']
train_x = train_data
train_x.to_csv('./data/train_x.csv', index=False)
train_y.to_csv('./data/train_y.csv', index=False)
test_data.to_csv('./data/test_x.csv', index=False)