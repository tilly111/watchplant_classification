import platform
import constants

import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import matplotlib
import matplotlib.pyplot as plt
from itertools import compress

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.feature_selection import RFECV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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

exp_names = ["Exp47_Ivy5"]  # "Exp44_Ivy2", "Exp45_Ivy4", "Exp46_Ivy0", "Exp47_Ivy5"
sensors = ["pn1", "pn3", "mu_ch1", "mu_ch2"]  # "pn1", "pn3", "mu_ch1", "mu_ch2"

feature_selection = "RFECV"  # "none" or "manual" or RFECV
manual_features_no = ["no_hjorth_comp", "no_wavelet_entropy"]  # list of manual selected features
manual_features_stim = ["stim_hjorth_comp", "stim_wavelet_entropy"]  # list of manual selected features

use_shap = True

for exp_name in exp_names:
    print(f"------------------- {exp_name} -------------------")

    # estimator to select
    # estimator = SVC(kernel="linear", probability=True, class_weight='balanced')  # , class_weight='balanced'
    estimator = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    # estimator = LinearDiscriminantAnalysis()

    # TODO make manual feature selection possible
    features = None
    all_feature_names = None
    for sensor in sensors:
        features_tmp = pd.read_csv(f"results/features_median_filter/{exp_name}_{sensor}_features.csv")
        if features is None:
            features = features_tmp
            all_feature_names = constants.NO_FEATURES
        else:
            features = pd.concat([features, features_tmp], axis=1)
            all_feature_names = all_feature_names + constants.NO_FEATURES
    print(f"raw data features: {features.shape}")
    # drop nan values of all rows
    features.dropna(axis=0, inplace=True)
    print(f"raw data after dropping nans: {features.shape}")
    if feature_selection == "manual":
        no = features[manual_features_no].to_numpy()
        stim = features[manual_features_stim].to_numpy()
        all_feature_names = manual_features_no * len(sensors)
    else:
        no = features[constants.NO_FEATURES].to_numpy()
        stim = features[constants.STIM_FEATURES].to_numpy()

    x_no = no
    y_no = np.zeros((no.shape[0],))
    x_stim = stim
    y_stim = np.ones((stim.shape[0],))

    x_train = np.concatenate((x_no, x_stim), axis=0)
    y_train = np.concatenate((y_no, y_stim), axis=0)

    # shuffle stuff and split into train and validation set; make reproducible by setting random state
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=14)

    min_max = MinMaxScaler()
    for i in range(x_train.shape[1]):  # min max scale all the features
        x_train[:, i] = min_max.fit_transform(np.reshape(x_train[:, i], (x_train[:, i].shape[0], 1))).squeeze()
        x_val[:, i] = min_max.transform(np.reshape(x_val[:, i], (x_val[:, i].shape[0], 1))).squeeze()
    print(f"x_train: {x_train.shape}, x_val: {x_val.shape}\ny_train: {y_train.shape}, y_val: {y_val.shape}")
    print(f"sum y_train {np.sum(y_train)}, sum y_val {np.sum(y_val)}")

    if feature_selection == "RFECV":
        selector = RFECV(estimator, step=1, cv=5, min_features_to_select=1)
        x_train = selector.fit_transform(x_train, y_train)
        x_val = selector.transform(x_val)
        all_feature_names = list(compress(all_feature_names, selector.support_))
    else:
        pass
    selector = estimator.fit(x_train, y_train)
    y_pred = selector.predict(x_val)

    if use_shap:
        number_of_features = len(all_feature_names)
        x_t = pd.DataFrame(data=x_val, columns=all_feature_names)
        x_train_wn = pd.DataFrame(data=x_train, columns=all_feature_names)

        explainer = shap.KernelExplainer(selector.predict_proba, x_train_wn)
        shap_value = explainer(x_t)
        # returns probability for class 0 and 1, but we only need one bc p = 1 - p
        shap_value.values = shap_value.values[:, :, 1]
        shap_value.base_values = shap_value.base_values[:, 1]

        print_frame = pd.DataFrame(data=np.zeros((1, number_of_features)), columns=all_feature_names)
        print_frame[:] = shap_value.abs.mean(axis=0).values
        # TODO does this makes sense
        print_frame = print_frame.iloc[:, :int(print_frame.shape[1]/len(sensors))]
        print("----------")
        for z in print_frame.columns:
            # print(f"{z}\t\t{print_frame[z].to_numpy().squeeze()}")
            if len(sensors) >= 2:
                test = print_frame[z].to_numpy().squeeze()
                print(f"{test}")  # {z}: \t
            else:
                print(f"{print_frame[z].to_numpy().squeeze()}")
        print("----------")
        shap.plots.bar(shap_value, max_display=10, show=False)

        # plt.figure()
        # shap.plots.bar(shap_value, max_display=number_of_features, show=True)
        # print(print_frame)
        shap_values = print_frame  # pd.concat([shap_values, print_frame], axis='index', ignore_index=True)
        # print(shap_values)

    print(f"accuracy: {accuracy_score(y_val, y_pred)}")
    print(f"F1-score: {f1_score(y_val, y_pred, average='weighted')}")
    print(f"confusion matrix: \n{confusion_matrix(y_val, y_pred)}")

    # tree_rules = export_text(selector, feature_names=all_feature_names)
    # print(tree_rules)
    plt.figure(figsize=(10, 10))
    plot_tree(selector, feature_names=all_feature_names, class_names=["no ozone", "ozone"], filled=True)
    plt.tight_layout()
    plt.savefig(f"results/dtc/{exp_name}_{sensors}_decision_tree.pdf")

    if x_train.shape[1] == 1:
        plt.figure(figsize=(10, 10))
        x_plot = np.linspace(0, 1, 1000)
        y_plot = selector.predict(x_plot.reshape(-1, 1))
        plt.scatter(x_plot[np.where(y_plot == 1)[0],], x_plot[np.where(y_plot == 1)[0],], c="red",
                    label="decision boundary",
                    alpha=0.1)
        plt.scatter(x_plot[np.where(y_plot == 0)[0],], x_plot[np.where(y_plot == 0)[0],], c="blue",
                    label="decision boundary",
                    alpha=0.1)
        plt.scatter(x_train[np.where(y_train == 1)[0], 0], x_train[np.where(y_train == 1)[0], 0], c="red", s=200,
                    label="train ozone")
        plt.scatter(x_train[np.where(y_train == 0)[0], 0], x_train[np.where(y_train == 0)[0], 0], c="cornflowerblue", s=200,
                    label="train  no ozone")
        plt.scatter(x_val[np.where(y_val == 1)[0], 0], x_val[np.where(y_val == 1)[0], 0], c="darkred", s=200,
                    label="validation ozone")
        plt.scatter(x_val[np.where(y_val == 0)[0], 0], x_val[np.where(y_val == 0)[0], 0], c="darkblue", s=200,
                    label="validation no ozone")
        plt.xlabel(manual_features_no[0])
        plt.ylabel(manual_features_no[0])
        plt.title(exp_names[0])
        plt.legend()
        plt.savefig(f"results/dtc/{exp_name}_{sensors}_decision_boundries.pdf")
    elif x_train.shape[1] == 2:
        # plt.figure(figsize=(30, 30))
        fig, axs = plt.subplots(1,1,figsize=(10, 10))
        DecisionBoundaryDisplay.from_estimator(selector, x_train, cmap=plt.cm.coolwarm, ax=axs)

        axs.scatter(x_train[np.where(y_train == 1)[0], 0], x_train[np.where(y_train == 1)[0], 1], c="red", s=200,
                    label="train ozone")
        axs.scatter(x_train[np.where(y_train == 0)[0], 0], x_train[np.where(y_train == 0)[0], 1], c="cornflowerblue",
                    s=200,
                    label="train  no ozone")
        axs.scatter(x_val[np.where(y_val == 1)[0], 0], x_val[np.where(y_val == 1)[0], 1], c="darkred", s=200,
                    label="validation ozone")
        axs.scatter(x_val[np.where(y_val == 0)[0], 0], x_val[np.where(y_val == 0)[0], 1], c="darkblue", s=200,
                    label="validation no ozone")
        axs.set_xlabel(manual_features_no[0])
        axs.set_ylabel(manual_features_no[1])
        axs.set_title(exp_names[0])
        axs.legend()
        plt.tight_layout()
        plt.savefig(f"results/dtc/{exp_name}_{sensors}_decision_boundries_two_features.pdf")
    else:
        print("no viz for room implemented")

    # plt.show()
