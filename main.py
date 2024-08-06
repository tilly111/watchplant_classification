import platform
import constants

import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from utils.learner_pipeline import get_pipeline_for_features
import matplotlib
import matplotlib.pyplot as plt
from itertools import compress

from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.feature_selection import RFECV, SequentialFeatureSelector, RFE, GenericUnivariateSelect
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from utils.feature_loader import load_tsfresh_feature, load_eddy_feature

import shap
from sklearn.tree import export_text
from sklearn.inspection import DecisionBoundaryDisplay

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

# total number of expositions: 152
# validate on the test data!
sensors = ["pn1"]  # "pn1", "pn3"
exp_names = ["Exp44_Ivy2", "Exp45_Ivy4", "Exp46_Ivy0", "Exp47_Ivy5"]
number_of_features = 787  # something between 1 and 787
number_of_repeats = 1  # usually 500
use_shap = True


if len(sensors) == 1 and sensors[0] == "pn1":
    data_pre_processor = None
    learner = RandomForestClassifier(criterion='entropy', max_features=7,
                                     min_samples_leaf=6,
                                     min_samples_split=14, n_estimators=512,
                                     warm_start=True)
    fc_df = pd.read_csv(f"/Volumes/Data/watchplant/Results/2024_felix/results/feature_combinations/feature_selection_results_{number_of_features}.csv")
    # select combo with the highest mean score
    max_combo = (fc_df[fc_df["score_mean"] == fc_df["score_mean"].max()]["combo"].values[0])[2:-2]
    max_combo = max_combo.split("', '")
elif len(sensors) == 1 and sensors[0] == "pn3":
    data_pre_processor = Normalizer()
    learner = ExtraTreesClassifier(bootstrap=True, criterion='entropy',
                                   max_features=0.7074865514350775,
                                   min_samples_leaf=2, min_samples_split=4,
                                   n_estimators=512, warm_start=True)
elif len(sensors) == 2 and sensors[0] == "pn1" and sensors[1] == "pn3":
    data_pre_processor = GenericUnivariateSelect(mode='fpr', param=0.3238036840257909)
    learner = HistGradientBoostingClassifier(early_stopping=False,
                                             l2_regularization=0.7089955744242014,
                                             learning_rate=0.3800630768981142,
                                             max_iter=512, max_leaf_nodes=77,
                                             min_samples_leaf=1,
                                             n_iter_no_change=1,
                                             validation_fraction=None,
                                             warm_start=True)


X_train, y_train = load_tsfresh_feature(exp_names, sensors, split=True, dir="")
X_test, y_test = load_tsfresh_feature(exp_names, sensors, split=True, dir="", test=True)

# select features
X_train = X_train[max_combo]
X_test = X_test[max_combo]
number_of_features = len(X_train.columns)

# classifier to select
pl_interpretable = get_pipeline_for_features(learner, data_pre_processor, X_train, y_train, list(X_train.columns))

acc_all = []
roc_all = []
f1_all = []
if use_shap:
    shap_values = pd.DataFrame(data=np.zeros((1, number_of_features)), columns=X_train.columns)
for i in range(number_of_repeats):
    trained = clone(pl_interpretable).fit(X_train.to_numpy(), y_train["y"])
    y_pred = trained.predict(X_test.to_numpy())
    y_pred_proba = trained.predict_proba(X_test.to_numpy())


    # print(f"accuracy: {accuracy_score(y_test, y_pred)}")
    acc_all.append(accuracy_score(y_test, y_pred))
    # print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba[:, 1])}")
    roc_all.append(roc_auc_score(y_test, y_pred_proba[:, 1]))
    # print(f"F1-score: {f1_score(y_test, y_pred, average='weighted')}")
    f1_all.append(f1_score(y_test, y_pred, average='weighted'))
    # print(f"confusion matrix: \n{confusion_matrix(y_test, y_pred)}")

    if use_shap:
        explainer = shap.KernelExplainer(trained.predict_proba, X_train)
        shap_value = explainer(X_test)

        # returns probability for class 0 and 1, but we only need one bc p = 1 - p
        shap_value.values = shap_value.values[:, :, 1]
        shap_value.base_values = shap_value.base_values[:, 1]


        shap_values[:] += shap_value.abs.mean(axis=0).values
        # print_frame = pd.DataFrame(data=np.zeros((1, number_of_features)), columns=X_train.columns)
        # print_frame[:] = shap_value.abs.mean(axis=0).values
        # print("----------")
        # for z in print_frame.columns:
        #     if print_frame[z].to_numpy().squeeze() > 0.0001:
        #         print(f"{z}: \t{print_frame[z].to_numpy().squeeze()}")
        # print("----------")
        # shap.plots.bar(shap_value, max_display=number_of_features, show=True)
        # # print(print_frame)
        # shap_values = pd.concat([shap_values, print_frame], axis='index', ignore_index=True)
        # # print(shap_values)


print("------------------------------------")
print(f"mean accuracy: {np.mean(acc_all)}")
print(f"mean ROC AUC: {np.mean(roc_all)}")
print(f"mean F1-score: {np.mean(f1_all)}")

if use_shap:
    shap_values = shap_values / number_of_repeats
    shap_values = shap_values.T
    shap_values = shap_values.rename(columns={0: "shap_values"})
    shap_values = shap_values.sort_values(by="shap_values", ascending=False)
    print(shap_values)
    print(X_test.columns)
    sv = shap.Explanation(values=shap_values["shap_values"].to_numpy(), feature_names=X_test.columns)
    # TODO does not work because shap_values not a explanantion object but a dataframe
    shap.plots.bar(sv, max_display=number_of_features, show=True)
    plt.show()

    # shap.plots.waterfall(shap_value[0], show=True)
    # shap.plots.beeswarm(shap_value, show=True)
    # shap.plots.force(shap_value, show=True)
    # shap.plots.heatmap(shap_value, show=True)
    # shap.plots.scatter(shap_value, show=True)
    # shap.plots.summary(shap_value, show=True)
    # shap.plots.dependence(shap_value, show=True)
    # shap.plots.pie(shap_value, show=True)
    # shap.plots.image_plot(shap_value, show=True)
    # shap.plots.decision(shap_value, show=True)
    # shap.plots.bar(shap_value, show=True)
    # shap.plots.text