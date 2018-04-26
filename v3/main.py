import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_survived_situation(coloumn_name):
    dead = train_data[coloumn_name][train_data.Survived == 0].value_counts()
    survived = train_data[coloumn_name][train_data.Survived == 1].value_counts()
    pd.DataFrame({'Dead': dead, 'Survived': survived}).plot(kind='bar')
    plt.title('Survived on' + coloumn_name)
    plt.ylabel('number of people')
    plt.xlabel(coloumn_name)


def deal_with_pclass(data):
    plot_survived_situation('Pclass')
    return data


def check_if_male(series):
    for x in series:
        x = 1 if x == 'male' else 0


def deal_with_sex(data):
    plot_survived_situation('Sex')
    data.Sex.apply(check_if_male)
    return data


def deal_with_age(data):
    # plot_survived_situation('Age')
    return data


def deal_with_sibsp(data):
    plot_survived_situation('SibSp')
    return data


def deal_with_parch(data):
    plot_survived_situation('Parch')
    return data


pd.set_option('expand_frame_repr', False)

train_data = pd.read_csv('../data/train.csv')
head = train_data.head(5)
print(head)

train_data = deal_with_pclass(train_data)
train_data = deal_with_sex(train_data)
train_data = deal_with_age(train_data)

head1 = train_data.head(5)
print(head1)

# plot_survived_situation('Parch')
# plot_survived_situation('Embarked')


# print(train_data.Cabin.value_counts())


# plt.show()
