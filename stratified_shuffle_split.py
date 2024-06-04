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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, SequentialFeatureSelector, RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from utils.feature_loader import load_tsfresh_feature, load_eddy_feature

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
elif platform.system() == "Windows":
    # TODO
    pass


# load data
exp_names = ["Exp44_Ivy2", "Exp45_Ivy4", "Exp46_Ivy0", "Exp47_Ivy5"]
sensors = ["pn1"]  # "pn1", "pn3", "mu_ch1", "mu_ch2"

x, y = load_tsfresh_feature(exp_names, sensors, clean=True)


# split data
sss = StratifiedShuffleSplit(n_splits=20, test_size=0.3, random_state=0)

roc_auc_list = []

# evaluate all splits
for i, (train_index, test_index) in enumerate(sss.split(x, y)):
    print(f"Fold {i}")
    # print(f"TRAIN: {train_index}, TEST: {test_index}")
    x_train, x_test = np.take(x, train_index, axis=0), np.take(x, test_index, axis=0)
    # x_train, x_test = x[train_index, :], x[test_index, :]
    # y_train, y_test = y[train_index], y[test_index]
    y_train, y_test = np.take(y, train_index, axis=0), np.take(y, test_index, axis=0)

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # train model TODO
    # clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    clf = RandomForestClassifier()
    # clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # evaluate model
    y_pred = clf.predict_proba(x_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)  # , pos_label=2
    roc_auc = auc(fpr, tpr)
    roc_auc_list.append(roc_auc)

    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc='lower right')
    # plt.show()
print(f"mean roc_auc: {np.mean(roc_auc_list)}")