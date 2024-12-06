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
import tikzplotlib
import matplotlib.pyplot as plt
from itertools import compress


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


# load data
pn1 = pd.read_csv("../results/acc_auc_testdata/acc_over_features_pn1_used_roc_auc.csv", header=None)
pn3 = pd.read_csv("../results/acc_auc_testdata/acc_over_features_pn3_used_roc_auc.csv", header=None)
pn13 = pd.read_csv("../results/acc_auc_testdata/acc_over_features_pn1_pn3_used_roc_auc.csv", header=None)

print(pn1.to_numpy().shape)

# plot
fig, ax = plt.subplots(figsize=(20, 4))
ax.plot(np.transpose(pn1.to_numpy()), label="Leaf", color="blue")
ax.plot(np.transpose(pn3.to_numpy()), label="Stem", color="orange")
ax.plot(np.transpose(pn13.to_numpy()), label="Combined", color="green")
ax.set_xlabel("Number of features")
ax.set_ylabel("Accuracy")
ax.set_ylim([0, 1])
plt.legend(loc="lower right")
plt.tight_layout()

ax.hlines(np.max(pn1.to_numpy()), 0, 1502, color="blue", linestyle="--")
ax.hlines(pn1.to_numpy()[0, 62 - 1], 0, 1502, color="blue", linestyle="--")
plt.text(x=1300, y=pn1.to_numpy()[0, 62 - 1] - 0.11, s=f'{np.max(np.max(pn1.to_numpy()) - pn1.to_numpy()[0, 62 - 1]):.4f}', color='blue', verticalalignment='bottom')

ax.hlines(np.max(pn3.to_numpy()), 0, 1502, color="orange", linestyle="--")
ax.hlines(pn3.to_numpy()[0, 69 - 1], 0, 1502, color="orange", linestyle="--")
plt.text(x=1300, y=pn3.to_numpy()[0, 69 - 1] - 0.11, s=f'{np.max(np.max(pn3.to_numpy()) - pn3.to_numpy()[0, 69 - 1]):.4f}', color='orange', verticalalignment='bottom')

ax.hlines(np.max(pn13.to_numpy()), 0, 1502, color="green", linestyle="--")
ax.hlines(pn13.to_numpy()[0, 94 - 1], 0, 1502, color="green", linestyle="--")
plt.text(x=1300, y=pn13.to_numpy()[0, 94 - 1] - 0.05, s=f'{np.max(np.max(pn13.to_numpy()) - pn13.to_numpy()[0, 94 - 1]):.4f}', color='green', verticalalignment='bottom')

# save
tikzplotlib.save("../plots/2024_felix/accuracy_over_features_used_roc_auc.tex")
plt.show()