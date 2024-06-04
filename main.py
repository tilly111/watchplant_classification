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
from sklearn.feature_selection import RFECV, SequentialFeatureSelector, RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

# total number of expositions: 152 ->

exp_names = ["Exp44_Ivy2", "Exp45_Ivy4", "Exp46_Ivy0", "Exp47_Ivy5"]  # "Exp44_Ivy2", "Exp45_Ivy4", "Exp46_Ivy0", "Exp47_Ivy5"
exp_names_train = ["Exp46_Ivy0"]
exp_names_val = ["Exp47_Ivy5"]
sensors = ["pn3"]  # "pn1", "pn3", "mu_ch1", "mu_ch2"

feature_selection = "RFE"  # "none" or "MANUAL" or "RFECV" or "SFS" or "RFE"
manual_features = [f'{sensors[0]}differential_potential_{sensors[0]}__linear_trend_timewise__attr_"slope"']
## exp 44
# manual_features = [f'pn1differential_potential_pn1__index_mass_quantile__q_0.6', 'pn1differential_potential_pn1__energy_ratio_by_chunks__num_segments_10__segment_focus_8']
# manual_features = [f'pn3differential_potential_pn3__index_mass_quantile__q_0.6', 'pn3differential_potential_pn3__energy_ratio_by_chunks__num_segments_10__segment_focus_8']
# manual_features = ['mu_ch1differential_potential_CH1__quantile__q_0.1', 'mu_ch1differential_potential_CH1__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.8']
# manual_features = ['mu_ch2differential_potential_CH2__quantile__q_0.1']
## exp 45
# manual_features = ['pn1differential_potential_pn1__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.4', 'pn1differential_potential_pn1__fft_coefficient__attr_"real"__coeff_6']
# manual_features = ['pn3differential_potential_pn3__mean_n_absolute_max__number_of_maxima_7']
# manual_features = ['mu_ch1differential_potential_CH1__fft_coefficient__attr_"abs"__coeff_0', 'mu_ch1differential_potential_CH1__permutation_entropy__dimension_4__tau_1']
# manual_features = ['mu_ch2differential_potential_CH2__linear_trend_timewise__attr_"slope"', 'mu_ch2differential_potential_CH2__fourier_entropy__bins_100']
## exp 46
# manual_features = ['pn1differential_potential_pn1__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.6', 'pn1differential_potential_pn1__fourier_entropy__bins_100']
# manual_features = ['pn3differential_potential_pn3__fft_coefficient__attr_"angle"__coeff_67', 'pn3differential_potential_pn3__agg_linear_trend__attr_"intercept"__chunk_len_50__f_agg_"min"', 'pn3differential_potential_pn3__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"mean"']
# manual_features = ['mu_ch1differential_potential_CH1__agg_linear_trend__attr_"intercept"__chunk_len_50__f_agg_"min"']
# manual_features = ['mu_ch2differential_potential_CH2__mean_n_absolute_max__number_of_maxima_7']
## exp 47
# manual_features = ['pn1differential_potential_pn1__change_quantiles__f_agg_"mean"__isabs_False__qh_0.6__ql_0.4', 'pn1differential_potential_pn1__approximate_entropy__m_2__r_0.3', 'pn1differential_potential_pn1__max_langevin_fixed_point__m_3__r_30']
# manual_features = ['pn3differential_potential_pn3__fft_coefficient__attr_"real"__coeff_98', 'pn3differential_potential_pn3__fft_coefficient__attr_"angle"__coeff_38']
# manual_features = ['mu_ch1differential_potential_CH1__mean_n_absolute_max__number_of_maxima_7']
# manual_features = ['mu_ch2differential_potential_CH2__linear_trend_timewise__attr_"slope"']

use_shap = True
scaling = "zscore"  # "none" or "min_max" or "zscore"

# for exp_name in exp_names:
#     print(f"------------------- {exp_name} -------------------")

# classifier to select
# estimator to select
estimator = SVC(kernel="linear", probability=True, class_weight='balanced')  # , class_weight='balanced'
# estimator = DecisionTreeClassifier(criterion="entropy", max_depth=5)
# estimator = LinearDiscriminantAnalysis()

# load feature data
x, y = load_tsfresh_feature(exp_names, sensors, clean=True)
x_train, y_train = load_tsfresh_feature(exp_names_train, sensors, clean=True)
x_val, y_val = load_tsfresh_feature(exp_names_val, sensors, clean=True)

# shuffle stuff and split into train and validation set; make reproducible by setting random state
# x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=14, shuffle=True)

# scale features
if scaling == "min_max":
    min_max = MinMaxScaler()
    for i in range(x_train.shape[1]):  # min max scale all the features
        x_train.iloc[:, i] = min_max.fit_transform(np.reshape(x_train.iloc[:, i], (x_train.iloc[:, i].shape[0], 1))).squeeze()
        x_val.iloc[:, i] = min_max.transform(np.reshape(x_val.iloc[:, i], (x_val.iloc[:, i].shape[0], 1))).squeeze()
elif scaling == "zscore":
    zscore = StandardScaler()
    for i in range(x_train.shape[1]):
        x_train.iloc[:, i] = zscore.fit_transform(np.reshape(x_train.iloc[:, i], (x_train.iloc[:, i].shape[0], 1))).squeeze()
        x_val.iloc[:, i] = zscore.transform(np.reshape(x_val.iloc[:, i], (x_val.iloc[:, i].shape[0], 1))).squeeze()
else:
    print("no scaling applied")
print(f"x_train: {x_train.shape}, x_val: {x_val.shape}\ny_train: {y_train.shape}, y_val: {y_val.shape}")
print(f"sum y_train {np.sum(y_train)}, sum y_val {np.sum(y_val)}")

# feature selection
if feature_selection == "RFECV":
    selector = RFECV(estimator, step=1, cv=5, min_features_to_select=1)
    x_train = selector.fit_transform(x_train, y_train)
    x_val = selector.transform(x_val)
    all_feature_names = list(compress(x.columns, selector.support_))
elif feature_selection == "SFS":
    selector = SequentialFeatureSelector(estimator, n_features_to_select=10, direction='backward')
    x_train = selector.fit_transform(x_train, y_train)
    x_val = selector.transform(x_val)
    all_feature_names = list(compress(x.columns, selector.support_))
    manual_features = all_feature_names
elif feature_selection == "RFE":
    selector = RFE(estimator, n_features_to_select=10, verbose=1)
    x_train = selector.fit_transform(x_train, y_train)
    x_val = selector.transform(x_val)
    all_feature_names = list(compress(x.columns, selector.support_))
    manual_features = all_feature_names
elif feature_selection == "MANUAL":
    x_train = x_train[manual_features]
    x_val = x_val[manual_features]
    all_feature_names = manual_features
else:
    all_feature_names = x.columns
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
        # if len(sensors) >= 2:
        #     test = print_frame[z].to_numpy().squeeze()
        #     print(f"{test}")  # {z}: \t
        # else:
        #     print(f"{print_frame[z].to_numpy().squeeze()}")
        if print_frame[z].to_numpy().squeeze() > 0.0001:
            print(f"{z}: \t{print_frame[z].to_numpy().squeeze()}")
    print("----------")
    shap.plots.bar(shap_value, max_display=10, show=False)
    plt.tight_layout()

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
# plt.figure(figsize=(10, 10))
# plot_tree(selector, feature_names=all_feature_names, class_names=["no ozone", "ozone"], filled=True)
# plt.tight_layout()
# plt.savefig(f"results/dtc/{exp_names}_{sensors}_decision_tree.pdf")

if x_train.shape[1] == 1:
    plt.figure(figsize=(10, 10))
    # x_plot = np.linspace(0, 1, 1000)
    # y_plot = selector.predict(x_plot.reshape(-1, 1))
    # plt.scatter(x_plot[np.where(y_plot == 1)[0],], x_plot[np.where(y_plot == 1)[0],], c="red",
    #             label="decision boundary",
    #             alpha=0.1)
    # plt.scatter(x_plot[np.where(y_plot == 0)[0],], x_plot[np.where(y_plot == 0)[0],], c="blue",
    #             label="decision boundary",
    #             alpha=0.1)
    plt.scatter(x_train.iloc[np.where(y_train == 1)[0], 0], np.zeros(x_train.iloc[np.where(y_train == 1)[0], 0].shape), c="red", s=200,
                label="train ozone")
    plt.scatter(x_train.iloc[np.where(y_train == 0)[0], 0], np.zeros(x_train.iloc[np.where(y_train == 0)[0], 0].shape), c="cornflowerblue", s=200,
                label="train  no ozone")
    plt.scatter(x_val.iloc[np.where(y_val == 1)[0], 0], np.zeros(x_val.iloc[np.where(y_val == 1)[0], 0].shape), c="darkred", s=200,
                label="validation ozone")
    plt.scatter(x_val.iloc[np.where(y_val == 0)[0], 0], np.zeros(x_val.iloc[np.where(y_val == 0)[0], 0].shape), c="darkblue", s=200,
                label="validation no ozone")
    plt.xlabel(manual_features[0])
    # plt.ylabel(manual_features[0])
    plt.title(exp_names[0])
    plt.legend()
    plt.savefig(f"results/dtc/{exp_names}_{sensors}_decision_boundries.pdf")
elif x_train.shape[1] == 2:
    # plt.figure(figsize=(30, 30))
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    DecisionBoundaryDisplay.from_estimator(selector, x_train, cmap=plt.cm.coolwarm, ax=axs)

    try:
        axs.scatter(x_train.iloc[np.where(y_train == 1)[0], 0], x_train.iloc[np.where(y_train == 1)[0], 1], c="red", s=200, label="train ozone")
    except:
        axs.scatter(x_train[np.where(y_train == 1)[0], 0], x_train[np.where(y_train == 1)[0], 1], c="red", s=200, label="train ozone")
    try:
        axs.scatter(x_train.iloc[np.where(y_train == 0)[0], 0], x_train.iloc[np.where(y_train == 0)[0], 1], c="cornflowerblue", s=200, label="train  no ozone")
    except:
        axs.scatter(x_train[np.where(y_train == 0)[0], 0], x_train[np.where(y_train == 0)[0], 1], c="cornflowerblue", s=200, label="train  no ozone")
    try:
        axs.scatter(x_val.iloc[np.where(y_val == 1)[0], 0], x_val.iloc[np.where(y_val == 1)[0], 1], c="darkred", s=200, label="validation ozone")
    except:
        axs.scatter(x_val[np.where(y_val == 1)[0], 0], x_val[np.where(y_val == 1)[0], 1], c="darkred", s=200, label="validation ozone")
    try:
        axs.scatter(x_val.iloc[np.where(y_val == 0)[0], 0], x_val.iloc[np.where(y_val == 0)[0], 1], c="darkblue", s=200, label="validation no ozone")
    except:
        axs.scatter(x_val[np.where(y_val == 0)[0], 0], x_val[np.where(y_val == 0)[0], 1], c="darkblue", s=200, label="validation no ozone")
    axs.set_xlabel(manual_features[0])
    axs.set_ylabel(manual_features[1])
    axs.set_title(f"val: {exp_names_val[0]}")
    axs.legend()
    plt.tight_layout()
    plt.savefig(f"results/dtc/{exp_names}_{sensors}_decision_boundries_two_features.pdf")
else:
    print(f"no viz for {x_train.shape[1]} dimensional feature space implemented")
    # print(f"features: {all_feature_names}")

plt.show()
