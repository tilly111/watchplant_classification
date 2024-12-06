import platform
import matplotlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing._data import Normalizer
from sklearn.feature_selection._univariate_selection import GenericUnivariateSelect

from utils.feature_loader import load_tsfresh_feature
import matplotlib.pyplot as plt

from utils.learner_pipeline import get_pipeline_for_features
from plotting_scripts.roc_curve_plotting import get_mccv_ROC_display, get_mccv_ROC_display_test

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
scoring = "roc_auc"

## select hyperparameters
roc_repeats = 500
# number_trees = 1000


for s in ["pn1", "pn3", "pn1_pn3"]:
    X_train, y_train = load_tsfresh_feature(exp_names, s, split=True, dir="")
    X_test, y_test = load_tsfresh_feature(exp_names, s, split=True, dir="", test=True)

    if s == "pn1":
        data_pre_processor = None
        learner = RandomForestClassifier(criterion='entropy', max_features=7,
                                         min_samples_leaf=6,
                                         min_samples_split=14, n_estimators=512,
                                         warm_start=True)
        number_of_features = 62
    elif s == "pn3":
        data_pre_processor = Normalizer()
        learner = ExtraTreesClassifier(bootstrap=True, criterion='entropy',
                                              max_features=0.7074865514350775,
                                              min_samples_leaf=2, min_samples_split=4,
                                              n_estimators=512, warm_start=True)
        number_of_features = 69
    elif s == "pn1_pn3":
        data_pre_processor = None
        learner = ExtraTreesClassifier(bootstrap=True, criterion='entropy',
                             max_features=0.1741962585712563,
                             min_samples_leaf=8, n_estimators=512,
                             warm_start=True)
        number_of_features = 94

    fc_df = pd.read_csv(
        f"/Volumes/Data/watchplant/Results/2024_felix/results/feature_combinations/{s}_feature_selection_results_{number_of_features}.csv")
    # select combo with the highest mean score
    max_combo = (fc_df[fc_df["score_mean"] == fc_df["score_mean"].max()]["combo"].values[0])[2:-2]
    max_combo = max_combo.split("', '")

    if s == "pn1_pn3":  # remove constant features for pn1 and pn3
        constant_columns = [col for col in X_train.columns if X_train[col].nunique() == 1]
        X_train.drop(columns=constant_columns, inplace=True)
        X_test.drop(columns=constant_columns, inplace=True)
    # select features
    X_train_tmp = pd.DataFrame(X_train[max_combo])
    X_test_tmp = pd.DataFrame(X_test[max_combo])


    pl_interpretable = get_pipeline_for_features(learner, data_pre_processor, X_train, y_train, list(X_train.columns))

    fig, axs = plt.subplots(1, 1, figsize=(7, 7))

    # mean_tpr, mean_fpr, save_acc = get_mccv_ROC_display(pl_interpretable, X_train, y_train, repeats=roc_repeats, ax=axs)
    mean_tpr, mean_fpr, save_acc = get_mccv_ROC_display_test(pl_interpretable, X_train_tmp, y_train, X_test_tmp, y_test, repeats=roc_repeats, ax=axs)

    # plt.savefig(f"plots/roc_curve_repeats_{roc_repeats}_{sensors}_all_features.png")
    print(f"Mean AUC: {save_acc.shape}")
    mean_fpr.to_csv(f"results/auc_curve/test_mean_fpr_{s}_auc_3.csv", index=True)
    mean_tpr.to_csv(f"results/auc_curve/test_mean_tpr_{s}_auc_3.csv", index=True)
    save_acc.to_csv(f"results/auc_curve/test_acc_{s}_auc_3.csv", index=True)

plt.show()