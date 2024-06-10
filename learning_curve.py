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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import RFECV, SequentialFeatureSelector, RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from utils.feature_loader import load_tsfresh_feature, load_eddy_feature


# for interactive plots
if platform.system() == "Darwin":
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    plt.rcParams.update({'font.size': 20})
    pd.set_option('display.max_rows', None)
elif platform.system() == "Linux":
    matplotlib.use('TkAgg')
elif platform.system() == "Windows":
    # TODO
    pass

# load data
exp_names = ["Exp44_Ivy2", "Exp45_Ivy4", "Exp46_Ivy0", "Exp47_Ivy5"]
sensors = ["pn1"]  # "pn1", "pn3", "mu_ch1", "mu_ch2"

x, y = load_tsfresh_feature(exp_names, sensors, clean=True)

num_splits = 10

print(type(x))
print(type(y))

x_0 = x.loc[y == 0]
x_1 = x.loc[y == 1]

print(x_0.shape, x_1.shape)


# print(x_train.shape, y_train.shape)
# print(y_train)

idxs = np.random.permutation(66)  # use for random sampling
idx_test = range(66, 76)

x_0_tmp = x_0.iloc[idx_test]
x_1_tmp = x_1.iloc[idx_test]
y_0_tmp = np.zeros((x_0_tmp.shape[0],))
y_1_tmp = np.ones((x_1_tmp.shape[0],))
x_test = pd.concat([x_0_tmp, x_1_tmp])
y_test = np.concatenate((y_0_tmp, y_1_tmp))

print(f"test: {x_test.shape}, {y_test.shape}")

data_set_size = []
roc_auc_list = []

for i in range(1, idxs.shape[0], 2):
    # split training data into two classes
    train_idxs = idxs[:i]
    x_0_tmp = x_0.iloc[train_idxs]
    x_1_tmp = x_1.iloc[train_idxs]
    y_0_tmp = np.zeros((x_0_tmp.shape[0],))
    y_1_tmp = np.ones((x_1_tmp.shape[0],))

    x_tmp = pd.concat([x_0_tmp, x_1_tmp])
    y_tmp = np.concatenate((y_0_tmp, y_1_tmp))

    # irgendwas mit classifier
    clf = ExtraTreesClassifier()
    clf.fit(x_tmp, y_tmp)
    y_pred = clf.predict_proba(x_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)  # , pos_label=2
    roc_auc = auc(fpr, tpr)
    print(f"training size: {i*2}")
    print(f"roc_auc: {roc_auc}")
    data_set_size.append(i*2)
    roc_auc_list.append(roc_auc)

plt.figure()
plt.plot(data_set_size, roc_auc_list, label="mean ROC AUC")
plt.xlabel("Number of samples")
plt.ylabel("ROC AUC")
plt.show()

