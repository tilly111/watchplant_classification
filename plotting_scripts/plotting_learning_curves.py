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
import tikzplotlib

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
    # plt.rcParams.update({'font.size': 20})
    pd.set_option('display.max_rows', None)
elif platform.system() == "Linux":
    # matplotlib.use('TkAgg')  # TODO removed because of server
    pass
elif platform.system() == "Windows":
    # TODO
    pass

# \definecolor{blue}{RGB}{0, 114, 178}
# \definecolor{orange}{RGB}{230, 159, 0}
# \definecolor{green}{RGB}{0, 158, 115}
colors = [(0/255, 114/255, 178/255), (230/255, 159/255, 0/255), (0/255, 158/255, 115/255)]

# local machine with hard drive mounted
if platform.system() == "Darwin":
    dir = "/Volumes/Data/watchplant/Results/2024_felix/"
    plotting = True
else:  # on server
    dir = "/abyss/home/code/watchplant_classification/"
    plotting = False

lcs = {}  # learning classifier system for each k
lcs[0] = pd.read_csv(f"{dir}results/lcs/pn1_lcs_62.csv")  # 62, 69, 94
lcs[1] = pd.read_csv(f"{dir}results/lcs/pn3_lcs_69.csv")
lcs[2] = pd.read_csv(f"{dir}results/lcs/pn1_pn3_lcs_94.csv")

# plot learning curves
fig, ax = plt.subplots(figsize=(16, 6))
# ax.plot(schedule, lc[0].mean(axis=1), label="train AUC")
for i, nfeat, ds, color in zip([0, 1, 2], [62,69,94], ["leaf", "stem", "combined"], colors):
    schedule, lc = [float(v) for v in lcs[i].columns], lcs[i].values
    if i == 1 or i == 2:
        schedule = [v * (122/152) for v in schedule]
    schedule = [int(v * 152) for v in schedule]

    mu = lc.mean(axis=0)
    std = lc.std(axis=0)
    ax.plot(schedule, mu, label=f"{ds}: {nfeat} features", color=color)
    ax.fill_between(schedule, mu - std*3, np.clip(mu + std*3, 0, 1), alpha=0.3, color=color)
ax.set_title(f"Learning Curves for Validation AUC ROC")
ax.legend()
ax.set_xlim([0, 152])
ax.set_ylabel("AUC ROC")
ax.set_xlabel("Number of samples")
# ax.set_ylim([0.45,0.8])
ax.axhline(1, color="black", linestyle="--")
ax.axhline(0.5, color="black", linestyle="--")

plt.tight_layout()
tikzplotlib.save("../plots/2024_felix/all_learning_curve_2.tex")

plt.show()
