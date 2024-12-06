import platform
import constants

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import tikzplotlib

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
pn1_acc = pd.read_csv("../results/auc_curve/test_acc_pn1_auc_3.csv", header=0)
pn3_acc = pd.read_csv("../results/auc_curve/test_acc_pn3_auc_3.csv", header=0)
pn1_pn3_acc = pd.read_csv("../results/auc_curve/test_acc_pn1_pn3_auc_3.csv", header=0)

pn1_acc.drop(columns=["Unnamed: 0"], inplace=True)
pn3_acc.drop(columns=["Unnamed: 0"], inplace=True)
pn1_pn3_acc.drop(columns=["Unnamed: 0"], inplace=True)

print(f" leaf mean acc: {pn1_acc.mean()}")
print(f" stem mean acc: {pn3_acc.mean()}")
print(f" combined mean acc: {pn1_pn3_acc.mean()}")


# plot roc curve
pn1_tpr = pd.read_csv("../results/auc_curve/test_mean_tpr_pn1_auc_3.csv", header=0)
pn1_fpr = pd.read_csv("../results/auc_curve/test_mean_fpr_pn1_auc_3.csv", header=0)
pn3_tpr = pd.read_csv("../results/auc_curve/test_mean_tpr_pn3_auc_3.csv", header=0)
pn3_fpr = pd.read_csv("../results/auc_curve/test_mean_fpr_pn3_auc_3.csv", header=0)
pn1_pn3_tpr = pd.read_csv("../results/auc_curve/test_mean_tpr_pn1_pn3_auc_3.csv", header=0)
pn1_pn3_fpr = pd.read_csv("../results/auc_curve/test_mean_fpr_pn1_pn3_auc_3.csv", header=0)

pn1_tpr.drop(columns=["Unnamed: 0"], inplace=True)
pn1_fpr.drop(columns=["Unnamed: 0"], inplace=True)
pn3_tpr.drop(columns=["Unnamed: 0"], inplace=True)
pn3_fpr.drop(columns=["Unnamed: 0"], inplace=True)
pn1_pn3_tpr.drop(columns=["Unnamed: 0"], inplace=True)
pn1_pn3_fpr.drop(columns=["Unnamed: 0"], inplace=True)

print(pn1_tpr.shape)
pn1_tpr["mean"] = pn1_tpr.mean(axis=1)
pn1_fpr["mean"] = pn1_fpr.mean(axis=1)
pn1_tpr["std"] = pn1_tpr.std(axis=1)
pn1_fpr["std"] = pn1_fpr.std(axis=1)

pn3_tpr["mean"] = pn3_tpr.mean(axis=1)
pn3_fpr["mean"] = pn3_fpr.mean(axis=1)
pn3_tpr["std"] = pn3_tpr.std(axis=1)
pn3_fpr["std"] = pn3_fpr.std(axis=1)

pn1_pn3_tpr["mean"] = pn1_pn3_tpr.mean(axis=1)
pn1_pn3_fpr["mean"] = pn1_pn3_fpr.mean(axis=1)
pn1_pn3_tpr["std"] = pn1_pn3_tpr.std(axis=1)
pn1_pn3_fpr["std"] = pn1_pn3_fpr.std(axis=1)

# getting the values
print(auc(pn1_fpr["trial_0"].values, pn1_tpr["mean"].values))
print(auc(pn3_fpr["trial_0"].values, pn3_tpr["mean"].values))
print(auc(pn1_pn3_fpr["trial_0"].values, pn1_pn3_tpr["mean"].values))


## generating the figure
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
# leaf
ax.fill_between(pn1_fpr["trial_0"], pn1_tpr["mean"]-pn1_tpr["std"], pn1_tpr["mean"] + pn1_tpr["std"], color='blue', alpha=.2)  # , label=r'$\pm$ 1 std. dev.'
ax.plot(pn1_fpr["trial_0"], pn1_tpr["mean"], color='blue', label="Leaf", lw=2, alpha=.8)
# stem
ax.fill_between(pn3_fpr["trial_0"], pn3_tpr["mean"]-pn3_tpr["std"], pn3_tpr["mean"] + pn3_tpr["std"], color='orange', alpha=.2)
ax.plot(pn3_fpr["trial_0"], pn3_tpr["mean"], color='orange', label="Stem", lw=2, alpha=.8)
# combined
ax.fill_between(pn1_pn3_fpr["trial_0"], pn1_pn3_tpr["mean"]-pn1_pn3_tpr["std"], pn1_pn3_tpr["mean"] + pn1_pn3_tpr["std"], color='green', alpha=.2)
ax.plot(pn1_pn3_fpr["trial_0"], pn1_pn3_tpr["mean"], color='green', label="Combined", lw=2, alpha=.8)

# Finalize plot
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Average ROC curve with variability')
ax.legend(loc="lower right")

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
tikzplotlib.save("../plots/2024_felix/roc_curve_all_test_feature_subset.tex")

plt.show()
