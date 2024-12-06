import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm


def get_mccv_ROC_display(learner, X, y, repeats, ax=None):
    # Initialize variables
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 101)

    save_tpr = pd.DataFrame(columns=[f"trial_{i}" for i in range(repeats)])
    save_fpr = pd.DataFrame(columns=[f"trial_{i}" for i in range(repeats)])
    save_acc = pd.DataFrame(columns=["accuracy"], index=range(repeats))

    # Create cross-validation strategy
    cv = StratifiedShuffleSplit(n_splits=repeats)

    # Loop over cross-validation splits
    pbar = tqdm(total=repeats)
    for train, test in cv.split(X, y):
        # Train the model
        classifier = clone(learner)
        y_score = classifier.fit(X.iloc[train], y.iloc[train].values.ravel()).predict_proba(X.iloc[test])

        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(y.iloc[test], y_score[:, -1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        save_tpr[f"trial_{pbar.n}"] = tprs[-1]
        save_fpr[f"trial_{pbar.n}"] = mean_fpr
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        save_acc["accuracy"].iloc[pbar.n] = accuracy_score(y.iloc[test], (y_score[:, -1] > 0.5).astype(int))
        pbar.update(1)
    pbar.close()

    # Calculate the mean and standard deviation of the true positive rates
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Plot the mean ROC curve
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    # Plot the standard deviation around the mean ROC curve
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    # Plot chance line
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    # Finalize plot
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Average ROC curve with variability')
    ax.legend(loc="lower right")

    return save_tpr, save_fpr, save_acc


def get_mccv_ROC_display_test(learner, X_train, y_train, X_test, y_test, repeats, ax=None):
    # Initialize variables
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 101)

    save_tpr = pd.DataFrame(columns=[f"trial_{i}" for i in range(repeats)])
    save_fpr = pd.DataFrame(columns=[f"trial_{i}" for i in range(repeats)])
    save_acc = pd.DataFrame(columns=["accuracy"], index=range(repeats))

    # Create cross-validation strategy
    cv = StratifiedShuffleSplit(n_splits=repeats)

    # Loop over cross-validation splits
    pbar = tqdm(total=repeats)
    for i in range(repeats):
    # for i_train, _ in cv.split(X_train, y_train):
        # Train the model
        classifier = clone(learner)
        y_score = classifier.fit(X_train, y_train.values.ravel()).predict_proba(X_test)
        # y_score = classifier.fit(X_train.iloc[i_train], y_train.iloc[i_train].values.ravel()).predict_proba(X_test)

        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(y_test, y_score[:, -1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        save_tpr[f"trial_{pbar.n}"] = tprs[-1]
        save_fpr[f"trial_{pbar.n}"] = mean_fpr
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        save_acc["accuracy"].iloc[pbar.n] = accuracy_score(y_test, (y_score[:, -1] > 0.5).astype(int))
        pbar.update(1)
    pbar.close()

    # Calculate the mean and standard deviation of the true positive rates
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Plot the mean ROC curve
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    # Plot the standard deviation around the mean ROC curve
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    # Plot chance line
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    # Finalize plot
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Average ROC curve with variability')
    ax.legend(loc="lower right")

    return save_tpr, save_fpr, save_acc
