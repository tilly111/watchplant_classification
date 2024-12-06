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


max_size = 1502
dir = "/Volumes/Data/watchplant/Results/2024_felix/"
sensors_names = "pn1_pn3"

dfs = {}
best_scores_per_k = []
best_combos_per_k = []
k_s = list(range(1, max_size + 1))

for k in range(1, max_size + 1):
    path = f"{dir}results/feature_combinations/{sensors_names}_feature_selection_results_{k}.csv"
    # path = f"{dir}results/feature_combinations/feature_selection_results_{k}.csv"
    if os.path.isfile(path):
        print(f"Loading {path}...")
        # dfs[k] = pd.read_csv(path)
        dfs_tmp = pd.read_csv(path)
        # dfs[k]["combo"] = [json.loads(e.replace("'", '"')) for e in dfs[k]["combo"]]
        dfs_tmp["combo"] = [json.loads(e.replace("'", '"')) for e in dfs_tmp["combo"]]
        # dfs[k]["scores"] = [json.loads(e) for e in dfs[k]["scores"]]
        dfs_tmp["scores"] = [json.loads(e) for e in dfs_tmp["scores"]]
        best_scores_per_k.append(dfs_tmp.iloc[0]["scores"])
        best_combos_per_k.append(dfs_tmp.iloc[0]["combo"])
        print(k, np.mean(best_scores_per_k[-1]), np.std(best_scores_per_k[-1]))

# for k in k_s:
#     df_fs = dfs[k]
#     best_scores_per_k.append(df_fs.iloc[0]["scores"])
#     best_combos_per_k.append(df_fs.iloc[0]["combo"])
#     print(k, np.mean(best_scores_per_k[-1]), np.std(best_scores_per_k[-1]))

fig, ax = plt.subplots(figsize=(10, 3))
mu = np.array([np.mean(v) for v in best_scores_per_k])
std = np.array([np.std(v) for v in best_scores_per_k])
print(std)
ax.plot(k_s, mu)
ax.fill_between(k_s, mu - std, np.clip(mu + std, 0, 1), alpha=0.2)
for k, combo, score in zip(k_s, best_combos_per_k, mu):
    print("Chosen feature combinations for", k, score, str(combo))  # , rotation=90)
ax.set_xlabel("Number of Features")
ax.set_ylabel("AUC ROC")
# ax.set_ylim([0.6, 0.8])
ax.axhline(max(mu), color="black", linestyle="--")
print(f"Best score {max(mu)} at {k_s[np.argmax(mu)]} features.")
tikzplotlib.save(f"plots/2024_felix/roc_auc_over_features_analysis_{sensors_names}.tex")
plt.show()