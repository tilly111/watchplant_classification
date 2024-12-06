import platform
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
from sklearn.feature_selection import RFECV, SequentialFeatureSelector, RFE, VarianceThreshold
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


max_size = 787
sensors_names = "pn3"
dfs = {}

best_scores_per_k = []
best_combos_per_k = []
for k in range(1, max_size + 1):
# for k in range(1, 3):
    path = f"/Volumes/Data/watchplant/Results/2024_felix/results/feature_combinations/{sensors_names}_feature_selection_results_{k}.csv"
    print(f"loading {path}...")

    dfs = pd.read_csv(path)
    # dfs[k]["combo"] = [json.loads(e.replace("'", '"')) for e in dfs[k]["combo"]]
    dfs["scores"] = [json.loads(e) for e in dfs["scores"]]
    best_scores_per_k.append(dfs.iloc[0]["scores"])

k_s = list(range(1, len(best_scores_per_k) + 1))
# for k in k_s:
#     df_fs = dfs[k]
#     best_scores_per_k.append(df_fs.iloc[0]["scores"])
#     # best_combos_per_k.append(df_fs.iloc[0]["combo"])
#     print(k, np.mean(best_scores_per_k[-1]), np.std(best_scores_per_k[-1]))



fig, ax = plt.subplots(figsize=(10, 3))
mu = np.array([np.mean(v) for v in best_scores_per_k])
std = np.array([np.std(v) for v in best_scores_per_k])
print(std)
ax.plot(k_s, mu)
ax.fill_between(k_s, mu - std, mu + std, alpha=0.2)
for k, score in zip(k_s, mu):
    print("Chosen feature combinations for", k, score)  # , rotation=90)
ax.set_xlabel("Number of Features")
ax.set_ylabel("AUC ROC")
# ax.set_ylim([0.6, 0.8])
ax.axhline(max(mu), color="black", linestyle="--")
print(f"Best score {max(mu)} at {k_s[np.argmax(mu)]} features.")
plt.show()