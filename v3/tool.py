import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_survived_situation_by_dataframe(df, title):
    dead = df[title][df.Survived == 0].value_counts()
    survived = df[title][df.Survived == 1].value_counts()
    pd.DataFrame({'Dead': dead, 'Survived': survived}).plot(kind='bar')
    plt.title('Survived on' + title)
    plt.ylabel('number of people')
    plt.xlabel(title)


def plot_survived_situation_of_cabin(df):
    x = pd.concat([pd.DataFrame(), df['Survived'], get_first_letter_and_set_N_when_value_is_nan(df)], axis=1)
    dead = x[0][x.Survived == 0].value_counts()
    survived = x[0][x.Survived == 1].value_counts()
    pd.DataFrame({'Dead': dead, 'Survived': survived}).plot(kind='bar')
    plt.title('Survived on Cabin')
    plt.ylabel('number of people')
    plt.xlabel('first letter of cabin')


def plot_picture(df):
    plot_survived_situation_by_dataframe(df, 'Pclass')
    plot_survived_situation_by_dataframe(df, 'Sex')
    plot_survived_situation_by_dataframe(df, 'Age')
    plot_survived_situation_by_dataframe(df, 'SibSp')
    plot_survived_situation_by_dataframe(df, 'Parch')
    plot_survived_situation_of_cabin(df)
    plot_survived_situation_by_dataframe(df, 'Embarked')

    plt.show()


def get_first_letter_and_set_N_when_value_is_nan(df):
    return pd.Series(['N' if pd.isnull(value) else value[0] for value in df['Cabin']])


def fill_missing_age_with_average(df):
    df['Age'] = df.Age.fillna(df.Age.mean())
    return df


def fill_missing_fare_with_average(df):
    df['Fare'] = df.Fare.fillna(df.Fare.mean())
    return df


def data_preparation(file_name):
    pd.set_option('expand_frame_repr', False)
    df = pd.read_csv(file_name)

    df = fill_missing_age_with_average(df)
    df = fill_missing_fare_with_average(df)
    # plot_picture(df)

    pclass = pd.get_dummies(df['Pclass'], prefix='Pclass')
    sex = pd.Series(np.where(df['Sex'] == 'male', 1, 0), name='Sex')
    age = df['Age']
    sibsp = pd.get_dummies(df['SibSp'], prefix='SibSp')
    parch = pd.get_dummies(df['Parch'], prefix='Parch')
    cabin = pd.get_dummies(get_first_letter_and_set_N_when_value_is_nan(df), prefix='Cabin')
    embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')

    train_x = pd.concat([pclass, sex, age, sibsp, parch, cabin, embarked], axis=1)
    train_y = None
    if 'Survived' in df.columns:
        train_y = df.Survived
    return train_x, train_y
