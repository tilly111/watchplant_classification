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
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.feature_selection import RFECV, SequentialFeatureSelector, RFE, GenericUnivariateSelect, VarianceThreshold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
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
sensors = ["pn1"]  # "pn1", "pn3", "bnb_no_wind"
exp_names = ["Exp44_Ivy2", "Exp45_Ivy4", "Exp46_Ivy0", "Exp47_Ivy5"]
# number_of_features = 1  # something between 1 and 787; None do best features
number_of_repeats = 500  # usually 500
use_shap = False
metric = "roc_auc"  # "accuracy", "roc_auc", "f1"

sensors_names = "_".join(sensors)

acc_over_features = []
acc_over_features_std = []
roc_over_features = []
roc_over_features_std = []
# feature_range = [1,2,3,4,5,6,7,8,9,10,62,69,94]  # 1,2,3,4,5,6,7,8,9,10,
# feature_range = range(1, 1502+1)  # 1502 TODO: run it for all possible combinations of features
feature_range = [62]
sklearn.set_config(enable_metadata_routing=True)
for number_of_features in feature_range:  # to automate a bit
    print(f"feat{number_of_features}------------------------------------")
    if len(sensors) == 1 and sensors[0] == "pn1":
        if metric == "accuracy":
            data_pre_processor = VarianceThreshold()
            learner = RandomForestClassifier(max_features=9, min_samples_leaf=11, min_samples_split=11, n_estimators=512,
                                            warm_start=True)

        elif metric == "roc_auc":
            data_pre_processor = None
            learner = RandomForestClassifier(criterion='entropy', max_features=7,
                                             min_samples_leaf=6,
                                             min_samples_split=14, n_estimators=512,
                                             warm_start=True)
        if number_of_features is None:
            # find best number of features
            max_acc = 0
            number_of_features = 1
            for i in range(1, 70):  # 62 is the best number of features
                tmp = pd.read_csv(f"/Volumes/Data/watchplant/Results/2024_felix/results/feature_combinations/feature_selection_results_{i}.csv")
                if tmp["score_mean"].max() > max_acc:
                    max_acc = tmp["score_mean"].max()
                    number_of_features = i
            print(f"best number of features: {number_of_features}")
        fc_df = pd.read_csv(f"/Volumes/Data/watchplant/Results/2024_felix/results/feature_combinations/{sensors_names}_feature_selection_results_{number_of_features}.csv")
        # select combo with the highest mean score
        max_combo = (fc_df[fc_df["score_mean"] == fc_df["score_mean"].max()]["combo"].values[0])[2:-2]
        max_combo = max_combo.split("', '")
    elif len(sensors) == 1 and sensors[0] == "pn3":
        if metric == "accuracy":
            data_pre_processor = MinMaxScaler()
            learner = KNeighborsClassifier(n_neighbors=27, weights='distance')
            # fc_df = pd.read_csv(
            #     f"/Volumes/Data/watchplant/Results/2024_felix/results/feature_combinations/pn3_feature_selection_results_{number_of_features}.csv")
            # # select combo with the highest mean score
            # max_combo = (fc_df[fc_df["score_mean"] == fc_df["score_mean"].max()]["combo"].values[0])[2:-2]
            # max_combo = max_combo.split("', '")
        elif metric == "roc_auc":
            data_pre_processor = Normalizer()
            learner = ExtraTreesClassifier(bootstrap=True, criterion='entropy', max_features=0.7074865514350775,
                                           min_samples_leaf=2, min_samples_split=4, n_estimators=512, warm_start=True)
        if number_of_features is None:
            # find best number of features
            max_acc = 0
            number_of_features = 1
            for i in range(1, 70):  # 69 is the best number of features
                tmp = pd.read_csv(
                    f"/Volumes/Data/watchplant/Results/2024_felix/results/feature_combinations/pn3_feature_selection_results_{i}.csv")
                if tmp["score_mean"].max() > max_acc:
                    max_acc = tmp["score_mean"].max()
                    number_of_features = i
                    print(f"new best with score: {max_acc} (n_features: {number_of_features})")
            print(f"best number of features: {number_of_features}")
        fc_df = pd.read_csv(
            f"/Volumes/Data/watchplant/Results/2024_felix/results/feature_combinations/pn3_feature_selection_results_{number_of_features}.csv")
        # select combo with the highest mean score
        max_combo = (fc_df[fc_df["score_mean"] == fc_df["score_mean"].max()]["combo"].values[0])[2:-2]
        max_combo = max_combo.split("', '")
    elif len(sensors) == 2 and sensors[0] == "pn1" and sensors[1] == "pn3":
        if metric == "accuracy":
            data_pre_processor = None
            learner = HistGradientBoostingClassifier(early_stopping=True,
                                                l2_regularization=5.617558408598586e-07,
                                                learning_rate=0.05202510577396622,
                                                max_iter=512,
                                                max_leaf_nodes=618,
                                                min_samples_leaf=17,
                                                n_iter_no_change=5,
                                                validation_fraction=0.029399129398777306,
                                                warm_start=True)
        elif metric == "roc_auc":
            data_pre_processor = None  # VarianceThreshold()
            learner = ExtraTreesClassifier(bootstrap=True, criterion='entropy', max_features=0.1741962585712563,
                                           min_samples_leaf=8, n_estimators=512, warm_start=True)

            # data_pre_processor = GenericUnivariateSelect(mode='fpr', param=0.3238036840257909)

        # learner = HistGradientBoostingClassifier(early_stopping=False,
        #                                          l2_regularization=0.7089955744242014,
        #                                          learning_rate=0.3800630768981142,
        #                                          max_iter=512, max_leaf_nodes=77,
        #                                          min_samples_leaf=1,
        #                                          n_iter_no_change=1,
        #                                          validation_fraction=None,
        #                                          warm_start=True)
        # learner = HistGradientBoostingClassifier(early_stopping=True,
        #                                l2_regularization=7.761316671313937e-08,
        #                                learning_rate=0.29462369916541875,
        #                                max_iter=512,
        #                                max_leaf_nodes=249,
        #                                min_samples_leaf=18,
        #                                n_iter_no_change=19,
        #                                validation_fraction=0.1873820076819333,
        #                                warm_start=True)
        fc_df = pd.read_csv(
            f"/Volumes/Data/watchplant/Results/2024_felix/results/feature_combinations/{sensors_names}_feature_selection_results_{number_of_features}.csv")
        # select combo with the highest mean score
        max_combo = (fc_df[fc_df["score_mean"] == fc_df["score_mean"].max()]["combo"].values[0])[2:-2]
        max_combo = max_combo.split("', '")
    elif len(sensors) == 1 and sensors[0] == "bnb_no_wind":
        data_pre_processor = None
        learner = ExtraTreesClassifier(max_features=0.5175976929395113,
                             min_samples_leaf=4, min_samples_split=11,
                             n_estimators=512, warm_start=True)
        X_train, y_train = load_tsfresh_feature(exp_names, sensors_names, split=True, dir="")
        max_combo = X_train.columns  # TODO what features to select

    X_train, y_train = load_tsfresh_feature(exp_names, sensors_names, split=True, dir="")
    X_test, y_test = load_tsfresh_feature(exp_names, sensors_names, split=True, dir="", test=True)

    if len(sensors) == 2:  # remove constant features for pn1 and pn3
        # print(f"shape before: {X_train.shape}")
        constant_columns = [col for col in X_train.columns if X_train[col].nunique() == 1]
        X_train.drop(columns=constant_columns, inplace=True)
        X_test.drop(columns=constant_columns, inplace=True)
        # print(f"Removing constant features (in total {len(constant_columns)} feature(s)).")
        # print(f"New shape: {X_train.shape}.")
        # max_combo = X_train.columns.tolist()[0]
        # print(max_combo)
        # exit(11)

    # select features
    X_train = pd.DataFrame(X_train[max_combo])
    X_test = pd.DataFrame(X_test[max_combo])

    number_of_features = len(X_train.columns)

    # print(f"number of features: {X_train.shape}")


    # classifier to select
    pl_interpretable = get_pipeline_for_features(learner, data_pre_processor, X_train, y_train, list(X_train.columns))

    acc_all = []
    roc_all = []
    f1_all = []
    if use_shap:
        shap_values = pd.DataFrame(data=np.zeros((1, number_of_features)), columns=X_train.columns)
    # for i in range(number_of_repeats):
    # Create cross-validation strategy
    cv = StratifiedShuffleSplit(n_splits=number_of_repeats)
    for train, test in cv.split(X_train, y_train):
        # print(f"repeat {i}")
        trained = clone(pl_interpretable).fit(X_train.iloc[train].values, y_train.iloc[train].values.ravel())
        y_pred = trained.predict(X_train.iloc[test])
        y_pred_proba = trained.predict_proba(X_train.iloc[test])


        # print(f"accuracy: {accuracy_score(y_test, y_pred)}")
        acc_all.append(accuracy_score(y_train.iloc[test], y_pred))
        # print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba[:, 1])}")
        roc_all.append(roc_auc_score(y_train.iloc[test], y_pred_proba[:, 1]))
        # print(f"F1-score: {f1_score(y_test, y_pred, average='weighted')}")
        f1_all.append(f1_score(y_train.iloc[test], y_pred, average='weighted'))
        # print(f"confusion matrix: \n{confusion_matrix(y_test, y_pred)}")

        if use_shap:
            explainer = shap.KernelExplainer(trained.predict_proba, X_train.iloc[train].values)
            shap_value = explainer(X_train.iloc[test])

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


    print(f"mean accuracy: {np.mean(acc_all):.4f} $\pm$ {np.std(acc_all):.4f}")  # \u00B1
    print(f"mean ROC AUC: {np.mean(roc_all):.4f} $\pm$ {np.std(roc_all):.4f}")
    print(f"mean F1-score: {np.mean(f1_all)}")
    acc_over_features.append(np.mean(acc_all))
    acc_over_features_std.append(np.std(acc_all))
    roc_over_features.append(np.mean(roc_all))
    roc_over_features_std.append(np.std(roc_all))

    if use_shap:
        shap_values = shap_values / number_of_repeats
        shap_values = shap_values.T
        shap_values = shap_values.rename(columns={0: "shap_values"})
        shap_values = shap_values.sort_values(by="shap_values", ascending=False)
        print(shap_values)
        shap_values.to_csv(f"results/shap/{sensors_names}_shap_values_{number_of_features}.csv")
        # print(X_test.columns)

        # this would be the plot for the shap values
        # sv = shap.Explanation(values=shap_values["shap_values"].to_numpy(), feature_names=X_test.columns)
        # # TODO does not work because shap_values not a explanantion object but a dataframe
        # shap.plots.bar(sv, max_display=number_of_features, show=True)
        # plt.show()

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

print(f"best accuracy: {max(acc_over_features)} reached with {feature_range[acc_over_features.index(max(acc_over_features))]} features")
print(f"best roc_auc: {max(roc_over_features)} reached with {feature_range[roc_over_features.index(max(roc_over_features))]} features")

acc_over_features_np = np.array(acc_over_features)
roc_over_features_np = np.array(roc_over_features)
acc_over_features_std_np = np.array(acc_over_features_std)
roc_over_features_std_np = np.array(roc_over_features_std)
acc_over_features_np.tofile(f"results/acc_auc_testdata/acc_over_features_{sensors_names}_used_{metric}_{number_of_repeats}.csv", sep=",")
roc_over_features_np.tofile(f"results/acc_auc_testdata/roc_over_features_{sensors_names}_used_{metric}_{number_of_repeats}.csv", sep=",")
acc_over_features_std_np.tofile(f"results/acc_auc_testdata/acc_over_features_std_{sensors_names}_used_{metric}_{number_of_repeats}.csv", sep=",")
roc_over_features_std_np.tofile(f"results/acc_auc_testdata/roc_over_features_std_{sensors_names}_used_{metric}_{number_of_repeats}.csv", sep=",")

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(feature_range, acc_over_features, label="accuracy")
axs[0].fill_between(feature_range, np.array(acc_over_features) - np.array(acc_over_features_std), np.array(acc_over_features) + np.array(acc_over_features_std), alpha=0.2)
axs[0].set_title("Accuracy")
axs[0].set_xlabel("Number of features")
axs[0].set_ylabel("Accuracy")

axs[1].plot(feature_range, roc_over_features, label="roc_auc")
axs[1].fill_between(feature_range, np.array(roc_over_features) - np.array(roc_over_features_std), np.array(roc_over_features) + np.array(roc_over_features_std), alpha=0.2)
axs[1].set_title("ROC AUC")
axs[1].set_xlabel("Number of features")
axs[1].set_ylabel("ROC AUC")
plt.show()