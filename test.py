import platform
import constants

import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import matplotlib
import matplotlib.pyplot as plt
from itertools import compress

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import shap
from sklearn.tree import export_text
from sklearn.inspection import DecisionBoundaryDisplay

# for interactive plots
if platform.system() == "Darwin":
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    plt.rcParams.update({'font.size': 20})
    pd.set_option('display.max_rows', None)
elif platform.system() == "Linux":
    matplotlib.use('TkAgg')


# total 788 features
f = pd.read_csv("results/features_tsfresh/Exp46_Ivy0_pn1_features.csv")
print(f.shape)
print(f.columns)

f_stim = f[f.columns[:788]]
f_no = f[f.columns[788:]]
f_stim.dropna(axis=1, inplace=True)
f_no.dropna(axis=1, inplace=True)


print(f_stim.shape)
print(f_no.shape)

print(f_stim.columns[233])
print(f_no.columns[233])

x_no = f_no
y_no = np.zeros((f_no.shape[0],))
x_stim = f_stim
y_stim = np.ones((f_stim.shape[0],))

x_train = np.concatenate((x_no, x_stim), axis=0)
y_train = np.concatenate((y_no, y_stim), axis=0)

# shuffle stuff and split into train and validation set; make reproducible by setting random state
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=14)

min_max = MinMaxScaler()
for i in range(x_train.shape[1]):  # min max scale all the features
    x_train[:, i] = min_max.fit_transform(np.reshape(x_train[:, i], (x_train[:, i].shape[0], 1))).squeeze()
    x_val[:, i] = min_max.transform(np.reshape(x_val[:, i], (x_val[:, i].shape[0], 1))).squeeze()
print(f"x_train: {x_train.shape}, x_val: {x_val.shape}\ny_train: {y_train.shape}, y_val: {y_val.shape}")
print(f"sum y_train {np.sum(y_train)}, sum y_val {np.sum(y_val)}")


estimator = DecisionTreeClassifier(criterion="entropy", max_depth=5)

# selector = RFECV(estimator, step=1, cv=5, min_features_to_select=1)
selector = SequentialFeatureSelector(estimator, n_features_to_select=10)
x_train = selector.fit_transform(x_train, y_train)
x_val = selector.transform(x_val)
all_feature_names = list(compress(f_stim.columns, selector.support_))

print(f"accuracy: {accuracy_score(y_val, y_val)}")
print(f"F1-score: {f1_score(y_val, y_val, average='weighted')}")
print(f"confusion matrix: \n{confusion_matrix(y_val, y_val)}")

print(f"selected features: {all_feature_names}")
print(f"number of selected features: {len(all_feature_names)}")