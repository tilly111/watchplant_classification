import platform
import matplotlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from utils.feature_loader import load_tsfresh_feature
import matplotlib.pyplot as plt

from utils.learner_pipeline import get_pipeline_for_features
from plotting_scripts.roc_curve_plotting import get_mccv_ROC_display

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


## load training/validation data: only for testing "Exp47_Ivy5"
exp_names = ["Exp44_Ivy2", "Exp45_Ivy4", "Exp46_Ivy0"]
sensors = ["pn1"]

X, y = load_tsfresh_feature(exp_names, sensors, clean=True)

## select hyperparameters
roc_repeats = 500
number_trees = 1000


learner = ExtraTreesClassifier(n_estimators=number_trees)
pl_interpretable = get_pipeline_for_features(learner, X, y, list(X.columns))

fig, axs = plt.subplots(1, 1, figsize=(7, 7))
get_mccv_ROC_display(pl_interpretable, X, y, repeats=roc_repeats, ax=axs)
plt.savefig(f"plots/roc_curve_repeats_{roc_repeats}_extra_tree_{number_trees}_all_features.pdf")
plt.show()