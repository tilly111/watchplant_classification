import platform
import matplotlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing._data import Normalizer
from sklearn.feature_selection._univariate_selection import GenericUnivariateSelect

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
exp_names = ["Exp44_Ivy2", "Exp45_Ivy4", "Exp46_Ivy0", "Exp47_Ivy5"]
sensors = ["pn1"]  # "pn1", "pn3", "mu_ch1", "mu_ch2"
scoring = "roc_auc"

X, y = load_tsfresh_feature(exp_names, sensors, split=True)

## select hyperparameters
roc_repeats = 500
number_trees = 1000

setting = pd.read_csv(f"results/naml_history_{scoring}_{sensors}.csv")
setting.sort_values(by=scoring, inplace=True, ascending=False)
pipeline = setting.iloc[0]["pipeline"]
print(pipeline)

# pn1
data_pre_processor = None
learner = RandomForestClassifier(criterion='entropy', max_features=7,
                                        min_samples_leaf=6,
                                        min_samples_split=14, n_estimators=1024,
                                        warm_start=True)
# pn3
# data_pre_processor = Normalizer()
# learner = ExtraTreesClassifier(bootstrap=True, criterion='entropy',
#                                       max_features=0.7074865514350775,
#                                       min_samples_leaf=2, min_samples_split=4,
#                                       n_estimators=512, warm_start=True)
# pn1 & pn3
# data_pre_processor = GenericUnivariateSelect(mode='fpr', param=0.3238036840257909)
# learner = HistGradientBoostingClassifier(early_stopping=False,
#                                                 l2_regularization=0.7089955744242014,
#                                                 learning_rate=0.3800630768981142,
#                                                 max_iter=512, max_leaf_nodes=77,
#                                                 min_samples_leaf=1,
#                                                 n_iter_no_change=1,
#                                                 validation_fraction=None,
#                                                 warm_start=True)

pl_interpretable = get_pipeline_for_features(learner, data_pre_processor, X, y, list(X.columns))

fig, axs = plt.subplots(1, 1, figsize=(7, 7))
get_mccv_ROC_display(pl_interpretable, X, y, repeats=roc_repeats, ax=axs)
plt.savefig(f"plots/roc_curve_repeats_{roc_repeats}_{sensors}_all_features.png")
plt.show()