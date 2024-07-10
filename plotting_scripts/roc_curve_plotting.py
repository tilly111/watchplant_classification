import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm


def get_mccv_ROC_display(learner, X, y, repeats, ax=None):
    # Initialize variables
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # Create cross-validation strategy
    n_splits = repeats
    cv = StratifiedShuffleSplit(n_splits=n_splits)

    # Loop over cross-validation splits
    pbar = tqdm(total=n_splits)
    for train, test in cv.split(X, y):
        # Train the model
        classifier = clone(learner)
        y_score = classifier.fit(X.iloc[train], y.iloc[train].values.ravel()).predict_proba(X.iloc[test])

        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(y.iloc[test], y_score[:, -1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
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

    # if fig is not None:
    #     plt.show()
