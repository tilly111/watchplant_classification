import platform
import constants

import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import matplotlib
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import shap

# for interactive plots
if platform.system() == "Darwin":
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    plt.rcParams.update({'font.size': 22})
    pd.set_option('display.max_rows', None)
elif platform.system() == "Linux":
    matplotlib.use('TkAgg')

exp_names = ["Exp44_Ivy2", "Exp45_Ivy4", "Exp46_Ivy0", "Exp47_Ivy5"]  # "Exp44_Ivy2", "Exp45_Ivy4", "Exp46_Ivy0", "Exp47_Ivy5"
sensors = ["mu_ch2", "mu_ch1"]  # "pn1", "pn3", "mu_ch1", "mu_ch2"
# shuffle stuff
# np.random.seed(10)  # TODO set to reproduce results


all_feature_names = None

# estimator = SVC(kernel="poly")  # , class_weight='balanced'
estimator = DecisionTreeClassifier(criterion="entropy", max_depth=5)
x_no, y_no = [], []
x_stim, y_stim = [], []
for exp_name in exp_names:
    print(f"------------------- {exp_name} -------------------")
    features = None
    for sensor in sensors:
        features_tmp = pd.read_csv(f"results/{exp_name}_{sensor}_features.csv")
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
    no = features[constants.NO_FEATURES].to_numpy()
    stim = features[constants.STIM_FEATURES].to_numpy()
    print(f"no: {no.shape}, stim: {stim.shape}")

    # x_no = no
    # y_no = np.zeros((no.shape[0],))
    # x_stim = stim
    # y_stim = np.ones((stim.shape[0],))
    x_no.append(no)
    y_no.append(np.zeros((no.shape[0],)))
    x_stim.append(stim)
    y_stim.append(np.ones((stim.shape[0],)))
    # else:
    #     x_no = np.concatenate((x_no, no), axis=0)
    #     y_no = np.concatenate((y_no, np.zeros((no.shape[0],))), axis=0)
    #     x_stim = np.concatenate((x_stim, stim), axis=0)
    #     y_stim = np.concatenate((y_stim, np.ones((stim.shape[0],))), axis=0)

x_train = np.concatenate(x_no+x_stim, axis=0)
y_train = np.concatenate(y_no+y_stim, axis=0)
print(f"x_train: {x_train.shape}, y_train: {x_train.shape}")

val_counter_c0 = int(x_train.shape[0] * 0.15)
val_counter_c1 = int(x_train.shape[0] * 0.15)
general_counter = 0
x_val = np.zeros((val_counter_c0 + val_counter_c1, x_train.shape[1]))
y_val = np.zeros((val_counter_c0 + val_counter_c1,))
while val_counter_c0 > 0:
    idx = np.random.randint(0, x_train.shape[0])
    if y_train[idx] == 0:
        x_val[general_counter, :] = x_train[idx, :]
        y_val[general_counter] = y_train[idx]
        val_counter_c0 -= 1
        general_counter += 1
        x_train = np.delete(x_train, idx, axis=0)
        y_train = np.delete(y_train, idx, axis=0)
while val_counter_c1 > 0:
    idx = np.random.randint(0, x_train.shape[0])
    if y_train[idx] == 1:
        x_val[general_counter, :] = x_train[idx, :]
        y_val[general_counter] = y_train[idx]
        val_counter_c1 -= 1
        general_counter += 1
        x_train = np.delete(x_train, idx, axis=0)
        y_train = np.delete(y_train, idx, axis=0)
# x_val = x_train
# y_val = y_train

min_max = MinMaxScaler()
for i in range(len(constants.NO_FEATURES)):  # min max scale all the features
    x_train[:, i] = min_max.fit_transform(np.reshape(x_train[:, i], (x_train[:, i].shape[0], 1))).squeeze()
    x_val[:, i] = min_max.transform(np.reshape(x_val[:, i], (x_val[:, i].shape[0], 1))).squeeze()
print(f"x_train: {x_train.shape}, x_val: {x_val.shape}\ny_train: {y_train.shape}, y_val: {y_val.shape}")
print(f"sum y_train {np.sum(y_train)}, sum y_val {np.sum(y_val)}")
selector = estimator.fit(x_train, y_train)
y_pred = selector.predict(x_val)

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
print("----------")
for z in print_frame.columns:
    # print(f"{z}\t\t{print_frame[z].to_numpy().squeeze()}")
    if len(sensors) == 2:
        test = print_frame[z].to_numpy().squeeze()
        print(f"{test[0]}")
    else:
        print(f"{print_frame[z].to_numpy().squeeze()}")
print("----------")

# plt.figure()
# shap.plots.bar(shap_value, max_display=number_of_features, show=True)
# print(print_frame)
shap_values = print_frame  # pd.concat([shap_values, print_frame], axis='index', ignore_index=True)
# print(shap_values)

print(f"accuracy: {accuracy_score(y_val, y_pred)}")
print(f"F1-score: {f1_score(y_val, y_pred, average='weighted')}")
print(f"confusion matrix: \n{confusion_matrix(y_val, y_pred)}")
