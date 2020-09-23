import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import timeit

from Adaline_class import Adaline_neuron

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data ',
                 header=None)

df[df[34] == '?'] = df[df[34] == '?'].replace({'?': '0'})
df[34] = df[34].astype(int)
df[1] = df[1].map({'N': -1, 'R': 1})

for col in df.columns:
    if col != 1:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

# print(df)

df_true = df[df[1] == 1]
df_false = df[df[1] == -1]

x_true = df_true.drop([1, 0, 2], axis=1)
y_true = df_true[1]

x_false = df_false.drop([1, 0, 2], axis=1)
y_false = df_false[1]
scores = []
cms = []
start = timeit.default_timer()
for i in [4, 8, 24]:  #
    x_train_true, x_test_true, y_train_true, y_test_true = train_test_split(x_true, y_true, test_size=0.3, random_state=i)
    x_train_false, x_test_false, y_train_false, y_test_false = train_test_split(x_false, y_false, test_size=0.3, random_state=i)
    x_train = x_train_true.append(x_train_false).reset_index().drop(["index"], axis=1)
    y_train = y_train_true.append(y_train_false).reset_index().drop(["index"], axis=1)
    x_test = x_test_true.append(x_test_false).reset_index().drop(["index"], axis=1)
    y_test = y_test_true.append(y_test_false).reset_index().drop(["index"], axis=1)
    model = Adaline_neuron(learningRate=0.05)
    for iteration in range(50):
        model.fit(x_train, y_train)
    # score1 = model.score(x_train, y_train)
    score = model.score(x_test, y_test)
    scores.append(score)
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    cms.append(cm)
# print(pd.crosstab(y_test, model.predict(x_test)))
stop = timeit.default_timer()
print('Adaline Train time ', stop - start, ' sec')
print(scores)
print(cms)
print("mean: ", np.mean(scores))
print("std: ", np.std(scores))

scores = []
cms = []
start = timeit.default_timer()
# https://www.kaggle.com/ahmethamzaemra/mlpclassifier-example

for i in [0, 15, 24]:
    x_train_true, x_test_true, y_train_true, y_test_true = train_test_split(x_true, y_true, test_size=0.3,
                                                                            random_state=i)
    x_train_false, x_test_false, y_train_false, y_test_false = train_test_split(x_false, y_false, test_size=0.3,
                                                                                random_state=i)
    x_train = x_train_true.append(x_train_false).reset_index().drop(["index"], axis=1)
    y_train = y_train_true.append(y_train_false).reset_index().drop(["index"], axis=1)
    x_test = x_test_true.append(x_test_false).reset_index().drop(["index"], axis=1)
    y_test = y_test_true.append(y_test_false).reset_index().drop(["index"], axis=1)
    clf = MLPClassifier(hidden_layer_sizes=(200, 100, 100), max_iter=700, alpha=0.002,
                        solver='sgd', verbose=10, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)
    cm = confusion_matrix(y_test, y_pred)
    cms.append(cm)

stop = timeit.default_timer()
print('Back propagation Train time ', stop - start, ' sec')
print(scores)
print(cms)
print("mean: ", np.mean(scores))
print("std: ", np.std(scores))




