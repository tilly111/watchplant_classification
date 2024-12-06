import platform
import random
import sys

import constants
import os
import json
from tqdm import tqdm

import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import matplotlib
import matplotlib.pyplot as plt
from itertools import compress

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing._data import Normalizer
from sklearn.feature_selection._univariate_selection import GenericUnivariateSelect
from sklearn.feature_selection import RFECV, SequentialFeatureSelector, RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import auc, get_scorer
from sklearn.base import clone

from utils.learner_pipeline import get_pipeline_for_features

from utils.feature_loader import load_tsfresh_feature, load_eddy_feature

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


# for interactive plots
if platform.system() == "Darwin":
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    plt.rcParams.update({'font.size': 20})
    pd.set_option('display.max_rows', None)
elif platform.system() == "Linux":
    # matplotlib.use('TkAgg')  # TODO removed because of server
    pass
elif platform.system() == "Windows":
    # TODO
    pass



data = pd.read_csv("/Volumes/Data/Nextcloud/CPS_share/projects/watchplant/data_sets/old/electro_potentials_ZZ_340_wtnbr/train_ANN.tsv", sep="\t", header=None)
data_test = pd.read_csv("/Volumes/Data/Nextcloud/CPS_share/projects/watchplant/data_sets/old/electro_potentials_ZZ_340_wtnbr/test_ANN.tsv", sep="\t", header=None)

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

X_test = data_test.iloc[:, 1:].values
y_test = data_test.iloc[:, 0].values

print(f"overview class distribution: {np.unique(y, return_counts=True)}")

# preprocessing
BPO_SCALING = 0.001  # impedance scaling that we have kilo ohm
BPO_OFFSET = 512000  # here we have no offset
X = (X - BPO_OFFSET) * BPO_SCALING
X_test = (X_test - BPO_OFFSET) * BPO_SCALING

# mean_c0 = np.mean(X[y == 0], axis=0)
# mean_c1 = np.mean(X[y == 1], axis=0)
# mean_c2 = np.mean(X[y == 2], axis=0)
# mean_c3 = np.mean(X[y == 3], axis=0)
# mean_c4 = np.mean(X[y == 4], axis=0)
#
# std_c0 = np.std(X[y == 0], axis=0)
# std_c1 = np.std(X[y == 1], axis=0)
# std_c2 = np.std(X[y == 2], axis=0)
# std_c3 = np.std(X[y == 3], axis=0)
# std_c4 = np.std(X[y == 4], axis=0)
#
# fig, axs = plt.subplots(5, 1, figsize=(10, 25))
# axs[0].plot(mean_c0, label="c0")
# axs[0].fill_between(range(len(mean_c0)), mean_c0 - std_c0, mean_c0 + std_c0, alpha=0.5)
# for i in np.random.random_integers(0, X[y == 0].shape[0], 10):
#     axs[0].plot(X[y == 0][i, :], alpha=0.6, color="gray")
# axs[0].set_title("wind")
# axs[0].set_xticks([])
#
# axs[1].plot(mean_c1, label="c1")
# axs[1].fill_between(range(len(mean_c1)), mean_c1 - std_c1, mean_c1 + std_c1, alpha=0.5)
# for i in np.random.random_integers(0, X[y == 1].shape[0], 10):
#     axs[1].plot(X[y == 1][i, :], alpha=0.6, color="gray")
# axs[1].set_title("temperature")
# axs[1].set_xticks([])
#
# axs[2].plot(mean_c2, label="c2")
# axs[2].fill_between(range(len(mean_c2)), mean_c2 - std_c2, mean_c2 + std_c2, alpha=0.5)
# for i in np.random.random_integers(0, X[y == 2].shape[0], 10):
#     axs[2].plot(X[y == 2][i, :], alpha=0.6, color="gray")
# axs[2].set_title("no stimulus")
# axs[2].set_xticks([])
#
# axs[3].plot(mean_c3, label="c3")
# axs[3].fill_between(range(len(mean_c3)), mean_c3 - std_c3, mean_c3 + std_c3, alpha=0.5)
# for i in np.random.random_integers(0, X[y == 3].shape[0], 10):
#     axs[3].plot(X[y == 3][i, :], alpha=0.6, color="gray")
# axs[3].set_title("blue light")
# axs[3].set_xticks([])
#
# axs[4].plot(mean_c4, label="c4")
# axs[4].fill_between(range(len(mean_c4)), mean_c4 - std_c4, mean_c4 + std_c4, alpha=0.5)
# for i in np.random.random_integers(0, X[y == 4].shape[0], 10):
#     axs[4].plot(X[y == 4][i, :], alpha=0.6, color="gray")
# axs[4].set_title("red light")
# # plt.tight_layout()
# plt.show()

## generate wind no
X_wind = X[y == 0]
X_wind_test = X_test[y_test == 0]
X_none = X[y == 2]
X_none_test = X_test[y_test == 2]

X_all = np.concatenate((X_none, X_wind), axis=0).transpose()
X_all_test = np.concatenate((X_none_test, X_wind_test), axis=0).transpose()
X_df = pd.DataFrame(X_all)
X_df_test = pd.DataFrame(X_all_test)

print(X_df.shape)

from tsfresh import extract_features
X_df_feature = None
X_df_feature_test = None
for i in range(X_df.shape[1]):
    X_tmp = X_df.iloc[:, i].to_frame()
    X_tmp.rename(columns={X_tmp.columns[0]: 'dp'}, inplace=True)
    X_tmp["id"] = np.ones((X_tmp.shape[0],))
    feat_tmp = extract_features(X_tmp, column_id="id", column_value=X_tmp.columns[0], n_jobs=1)
    if X_df_feature is not None:
        X_df_feature = pd.concat([X_df_feature, feat_tmp], axis=0)

    else:
        X_df_feature = feat_tmp


for i in range(X_df_test.shape[1]):
    X_tmp_test = X_df_test.iloc[:, i].to_frame()
    X_tmp_test.rename(columns={X_tmp_test.columns[0]: 'dp'}, inplace=True)
    X_tmp_test["id"] = np.ones((X_tmp_test.shape[0],))
    feat_tmp_test = extract_features(X_tmp_test, column_id="id", column_value=X_tmp_test.columns[0], n_jobs=1)
    if X_df_feature_test is not None:
        X_df_feature_test = pd.concat([X_df_feature_test, feat_tmp_test], axis=0)
    else:
        X_df_feature_test = feat_tmp_test

print(X_df_feature.shape)
y = np.concatenate([np.zeros((X_none.shape[0],)), np.ones((X_wind.shape[0],))], axis=0)
y_test = np.concatenate([np.zeros((X_none_test.shape[0],)), np.ones((X_wind_test.shape[0],))], axis=0)
y = pd.DataFrame(y, columns=["y"])
y_test = pd.DataFrame(y_test, columns=["y"])
print(y.shape)

# X_df_feature.to_csv("data_preprocessed/X_bnb_no_wind_tmp.csv", index=False)
# y.to_csv("data_preprocessed/y_bnb_no_wind_tmp.csv", index=False)
# X_df_feature = pd.read_csv("data_preprocessed/X_bnb_no_wind_tmp.csv")
# y = pd.read_csv("data_preprocessed/y_bnb_no_wind_tmp.csv")

# drop only nan columns (not rows)
X_df_feature.dropna(axis=1, how='all', inplace=True)
X_df_feature_test.dropna(axis=1, how='all', inplace=True)
# drop constant features
constant_columns = [col for col in X_df_feature.columns if X_df_feature[col].nunique() == 1]
X_df_feature.drop(columns=constant_columns, inplace=True)
X_df_feature_test.drop(columns=constant_columns, inplace=True)
print(f"Removing constant features (in total {len(constant_columns)} feature(s)).")
print(f"New shape: {X_df_feature.shape}.")

# splitting the data
rows_to_remove = np.where(X_df_feature.isna().any(axis=1))[0]
y.drop(index=rows_to_remove, inplace=True)
X_df_feature.dropna(inplace=True)

rows_to_remove = np.where(X_df_feature_test.isna().any(axis=1))[0]
y_test.drop(index=rows_to_remove, inplace=True)
X_df_feature_test.dropna(inplace=True)

# print(X_df_feature.shape, y.shape)
#
# X_train, X_test, y_train, y_test = train_test_split(X_df_feature, y, test_size=0.2, random_state=0)
#
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# save the data to csv
X_df_feature.to_csv(f"data_preprocessed/split_data/X_train_bnb_no_wind.csv", index=False)
X_df_feature_test.to_csv(f"data_preprocessed/split_data/X_test_bnb_no_wind.csv", index=False)
y.to_csv(f"data_preprocessed/split_data/y_train_bnb_no_wind.csv", index=False)
y_test.to_csv(f"data_preprocessed/split_data/y_test_bnb_no_wind.csv", index=False)
