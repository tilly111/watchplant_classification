import platform
import os
import constants

import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib
import matplotlib.pyplot as plt
import tikzplotlib

from preprocessing.features import calc_all_features, calc_ts_fresh

# for interactive plots
if platform.system() == "Darwin":
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    plt.rcParams.update({'font.size': 22})
    # pd.set_option('display.max_rows', None)
elif platform.system() == "Linux":
    matplotlib.use('TkAgg')


def cut_data(df, begin, end):
    df = df.loc[df.index >= begin]
    df = df.loc[df.index <= end]
    return df


# exp_names = ["Exp44_Ivy2", "Exp45_Ivy4", "Exp46_Ivy0", "Exp47_Ivy5"]  # , "Exp45_Ivy4", "Exp46_Ivy0", "Exp47_Ivy5"
# plotting = False
# # if external drive is mounted:
# # DIR = f"/Volumes/Data/watchplant/gas_experiments/ozone_cut/{exp_name}"
# # else
# # DIR = f"data/gas_experiments/ozone_cut/{exp_name}"
# no_pn_1_all = np.empty((0, 1201))
# no_pn_3_all = np.empty((0, 1201))
# stim_pn_1_all = np.empty((0, 1201))
# stim_pn_3_all = np.empty((0, 1201))
#
# for exp_name in exp_names:
#     print(f"experiment {exp_name}")
#     dir_path = "data"  # or "/Volumes/Data/watchplant"
#     DIR = f"{dir_path}/gas_experiments/ozone_cut/{exp_name}"
#     experiments = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
#     timestamps = pd.read_csv(f"{dir_path}/gas_experiments/ozone/{exp_name}/times.csv")
#
#     for i, ts in enumerate(timestamps["times"]):
#         print(f"experiment number {i}")
#         # get timings
#         stimulus_application_begin = pd.to_datetime(ts, format="%Y-%m-%d %H:%M:%S")
#         stimulus_application_end = stimulus_application_begin + timedelta(minutes=10)  # 10 min stimulus
#
#         # load data
#         path = os.path.join(DIR, f"experiment_{i}.csv")
#         df = pd.read_csv(path, index_col=["timestamp"], date_format="%Y-%m-%d %H:%M:%S.%f")
#
#         # get ozone
#         df_ozone = df[["O3_1", "O3_2"]]
#         df_ozone = df_ozone.dropna()
#
#         # drop unnecessary values
#         df_pn_1 = df[["differential_potential_pn1"]]
#         df_pn_3 = df[["differential_potential_pn3"]]
#         df_pn_1 = df_pn_1.dropna()
#         df_pn_3 = df_pn_3.dropna()
#         if df_pn_1.empty:
#             print(f"pn 1 empty in experiment {i}")
#         if df_pn_3.empty:
#             print(f"pn 3 empty in experiment {i}")
#
#         Vref = 2.5
#         Gain = 4
#         databits = 8388608
#         df_pn_1["differential_potential_pn1"] = ((df_pn_1[
#                                                       "differential_potential_pn1"] / databits) - 1) * Vref / Gain * 1000
#         df_pn_3["differential_potential_pn3"] = ((df_pn_3[
#                                                       "differential_potential_pn3"] / databits) - 1) * Vref / Gain * 1000
#
#         # EMD filter data
#         # df_mu = emd_filter(df_mu, c=2, verbose=plotting)
#         # df_pn_1 = emd_filter(df_pn_1, c=6, verbose=plotting)
#         # if not df_pn_3.empty:
#         #     df_pn_3 = emd_filter(df_pn_3, c=6, verbose=plotting)
#
#         # some static filter
#         df_pn_1.drop(df_pn_1[df_pn_1["differential_potential_pn1"] >= 200].index, inplace=True)
#         df_pn_1.drop(df_pn_1[df_pn_1["differential_potential_pn1"] <= -200].index, inplace=True)
#         df_pn_3.drop(df_pn_3[df_pn_3["differential_potential_pn3"] >= 200].index, inplace=True)
#         df_pn_3.drop(df_pn_3[df_pn_3["differential_potential_pn3"] <= -200].index, inplace=True)
#
#         # filter stuff
#         N = 10
#         df_pn_1["differential_potential_pn1"] = df_pn_1["differential_potential_pn1"].rolling(window=N).median()
#         df_pn_3["differential_potential_pn3"] = df_pn_3["differential_potential_pn3"].rolling(window=N).median()
#
#         # downsample the signal
#         df_pn_1 = df_pn_1.resample('0.5S').mean()
#         df_pn_3 = df_pn_3.resample('0.5S').mean()
#
#         # cut data into time slices
#         bg_start = stimulus_application_begin - timedelta(minutes=20)
#         bg_end = stimulus_application_begin - timedelta(minutes=10)
#         no_start = stimulus_application_begin - timedelta(minutes=10)
#         no_end = stimulus_application_begin
#
#         if not df_pn_1.empty:
#             bg_pn_1 = cut_data(df_pn_1, bg_start, bg_end)
#             no_pn_1 = cut_data(df_pn_1, no_start, no_end)
#             stim_pn_1 = cut_data(df_pn_1, stimulus_application_begin, stimulus_application_end)
#
#         if not df_pn_3.empty:
#             bg_pn_3 = cut_data(df_pn_3, bg_start, bg_end)
#             no_pn_3 = cut_data(df_pn_3, no_start, no_end)
#             stim_pn_3 = cut_data(df_pn_3, stimulus_application_begin, stimulus_application_end)
#
#         ## scale the data
#         # get the max and min value of the channel for normalization
#         print(bg_pn_1.shape, no_pn_1.shape, stim_pn_1.shape)
#
#         if bg_pn_1.shape[0] == 1201 and no_pn_1.shape[0] == 1201 and stim_pn_1.shape[0] == 1201:
#             no = no_pn_1["differential_potential_pn1"].values - bg_pn_1["differential_potential_pn1"].values
#             stim = stim_pn_1["differential_potential_pn1"].values - bg_pn_1["differential_potential_pn1"].values
#             no = no.reshape(1, -1)
#             stim = stim.reshape(1, -1)
#
#             no_pn_1_all = np.concatenate((no_pn_1_all, no), axis=0)
#             stim_pn_1_all = np.concatenate((stim_pn_1_all, stim), axis=0)
#         if bg_pn_3.shape[0] == 1201 and no_pn_3.shape[0] == 1201 and stim_pn_3.shape[0] == 1201:
#             no = no_pn_3["differential_potential_pn3"].values - bg_pn_3["differential_potential_pn3"].values
#             stim = stim_pn_3["differential_potential_pn3"].values - bg_pn_3["differential_potential_pn3"].values
#             no = no.reshape(1, -1)
#             stim = stim.reshape(1, -1)
#
#             no_pn_3_all = np.concatenate((no_pn_3_all, no), axis=0)
#             stim_pn_3_all = np.concatenate((stim_pn_3_all, stim), axis=0)
#
#         # plot slices
#         if plotting:
#             fig, ax1 = plt.subplots(figsize=(20, 10))
#             ax1.axvspan(stimulus_application_begin, stimulus_application_end, facecolor='0.2', alpha=0.5)
#             ax1.plot(no_pn_1.index, no_pn_1["differential_potential_pn1"], label="pn1")
#             ax1.plot(stim_pn_1.index, stim_pn_1["differential_potential_pn1"], label="pn1")
#
#             plt.show()
#
# np.save("results/threshold_model/no_pn1.npy", no_pn_1_all)
# np.save("results/threshold_model/stim_pn1.npy", stim_pn_1_all)
# np.save("results/threshold_model/no_pn3.npy", no_pn_3_all)
# np.save("results/threshold_model/stim_pn3.npy", stim_pn_3_all)


# # load data
no_pn1 = np.load("results/threshold_model/no_pn1.npy")
stim_pn1 = np.load("results/threshold_model/stim_pn1.npy")
no_pn3 = np.load("results/threshold_model/no_pn3.npy")
stim_pn3 = np.load("results/threshold_model/stim_pn3.npy")

thresholds = np.linspace(-10, 10, 401)
# accuracy_pn1 = np.zeros((len(thresholds),))
# accuracy_pn3 = np.zeros((len(thresholds),))

best_acc_pn1 = []
best_acc_pn3 = []
best_acc_mean = []

for k in range(500):
    print(f"iteration {k}")
    # for saving the results
    pn_1_tp = np.zeros((len(thresholds),))
    pn_1_tn = np.zeros((len(thresholds),))
    pn_1_fp = np.zeros((len(thresholds),))
    pn_1_fn = np.zeros((len(thresholds),))

    pn_3_tp = np.zeros((len(thresholds),))
    pn_3_tn = np.zeros((len(thresholds),))
    pn_3_fp = np.zeros((len(thresholds),))
    pn_3_fn = np.zeros((len(thresholds),))

    # split data randomly into train and test
    no_pn1_train_idx = np.random.choice(no_pn1.shape[0], int(no_pn1.shape[0] * 0.8), replace=False)
    no_pn1_test_idx = np.setdiff1d(np.arange(no_pn1.shape[0]), no_pn1_train_idx)
    no_pn3_train_idx = np.random.choice(no_pn3.shape[0], int(no_pn3.shape[0] * 0.8), replace=False)
    no_pn3_test_idx = np.setdiff1d(np.arange(no_pn3.shape[0]), no_pn3_train_idx)

    stim_pn1_train_idx = np.random.choice(stim_pn1.shape[0], int(stim_pn1.shape[0] * 0.8), replace=False)
    stim_pn1_test_idx = np.setdiff1d(np.arange(stim_pn1.shape[0]), stim_pn1_train_idx)
    stim_pn3_train_idx = np.random.choice(stim_pn3.shape[0], int(stim_pn3.shape[0] * 0.8), replace=False)
    stim_pn3_test_idx = np.setdiff1d(np.arange(stim_pn3.shape[0]), stim_pn3_train_idx)

    # training loop
    for t, threshold in enumerate(thresholds):
        for i in no_pn1_train_idx:
            if np.any(no_pn1[i] >= threshold):
                pn_1_fp[t] += 1
            else:
                pn_1_tn[t] += 1
        for i in stim_pn1_train_idx:
            if np.any(stim_pn1[i] >= threshold):
                pn_1_tp[t] += 1
            else:
                pn_1_fn[t] += 1

        for i in no_pn3_train_idx:
            if np.any(no_pn3[i] >= threshold):
                pn_3_fp[t] += 1
            else:
                pn_3_tn[t] += 1
        for i in stim_pn3_train_idx:
            if np.any(stim_pn3[i] >= threshold):
                pn_3_tp[t] += 1
            else:
                pn_3_fn[t] += 1
        # print(f"threshold: {threshold}")
        # print(f"acc: {(pn_1_tp[t] + pn_1_tn[t]) / (pn_1_tp[t] + pn_1_tn[t] + pn_1_fp[t] + pn_1_fn[t])}")
    accuracy_pn1 = (pn_1_tp + pn_1_tn) / (pn_1_tp + pn_1_tn + pn_1_fp + pn_1_fn)
    accuracy_pn3 = (pn_3_tp + pn_3_tn) / (pn_3_tp + pn_3_tn + pn_3_fp + pn_3_fn)
    accuracy_mean = (pn_1_tp + pn_1_tn + pn_3_tp + pn_3_tn) / (pn_1_tp + pn_1_tn + pn_1_fp + pn_1_fn + pn_3_tp + pn_3_tn + pn_3_fp + pn_3_fn)

    best_threshold_pn1 = thresholds[np.argmax(accuracy_pn1)]
    best_threshold_pn3 = thresholds[np.argmax(accuracy_pn3)]
    best_threshold_mean = thresholds[np.argmax(accuracy_mean)]

    # print(f"best threshold pn1: {best_threshold_pn1}")
    # print(f"best threshold pn3: {best_threshold_pn3}")
    # print(f"best threshold mean: {best_threshold_mean}")

    # test loop
    pn_1_tp = 0
    pn_1_tn = 0
    pn_1_fp = 0
    pn_1_fn = 0
    pn_3_tn = 0
    pn_3_tp = 0
    pn_3_fp = 0
    pn_3_fn = 0

    for i in no_pn1_test_idx:
        if np.any(no_pn1[i] >= best_threshold_pn1):
            pn_1_fp += 1
        else:
            pn_1_tn += 1
    for i in stim_pn1_train_idx:
        if np.any(stim_pn1[i] >= best_threshold_pn1):
            pn_1_tp += 1
        else:
            pn_1_fn += 1

    for i in no_pn3_train_idx:
        if np.any(no_pn3[i] >= best_threshold_pn3):
            pn_3_fp += 1
        else:
            pn_3_tn += 1
    for i in stim_pn3_train_idx:
        if np.any(stim_pn3[i] >= best_threshold_pn3):
            pn_3_tp += 1
        else:
            pn_3_fn += 1

    # print(f"pn1 test accuracy: {(pn_1_tp + pn_1_tn) / (pn_1_tp + pn_1_tn + pn_1_fp + pn_1_fn)}")
    # print(f"pn3 test accuracy: {(pn_3_tp + pn_3_tn) / (pn_3_tp + pn_3_tn + pn_3_fp + pn_3_fn)}")
    # print(f"mean test accuracy {(pn_1_tp + pn_1_tn + pn_3_tp + pn_3_tn) / (pn_1_tp + pn_1_tn + pn_1_fp + pn_1_fn + pn_3_tp + pn_3_tn + pn_3_fp + pn_3_fn)}")
    best_acc_pn1.append((pn_1_tp + pn_1_tn) / (pn_1_tp + pn_1_tn + pn_1_fp + pn_1_fn))
    best_acc_pn3.append((pn_3_tp + pn_3_tn) / (pn_3_tp + pn_3_tn + pn_3_fp + pn_3_fn))
    best_acc_mean.append((pn_1_tp + pn_1_tn + pn_3_tp + pn_3_tn) / (pn_1_tp + pn_1_tn + pn_1_fp + pn_1_fn + pn_3_tp + pn_3_tn + pn_3_fp + pn_3_fn))

print(f"best accuracy pn1: {np.mean(best_acc_pn1)}")
print(f"best accuracy pn3: {np.mean(best_acc_pn3)}")
print(f"best accuracy mean: {np.mean(best_acc_mean)}")
# fig, axs = plt.subplots(2, 1, figsize=(10, 20))
# axs[0].plot(thresholds, pn_1_tp, label="tp")
# axs[0].plot(thresholds, pn_1_tn, label="tn")
# axs[0].plot(thresholds, pn_1_fp, label="fp")
# axs[0].plot(thresholds, pn_1_fn, label="fn")
# axs[0].legend()
# axs[0].set_title("pn1")
#
# axs[1].plot(thresholds, pn_3_tp, label="tp")
# axs[1].plot(thresholds, pn_3_tn, label="tn")
# axs[1].plot(thresholds, pn_3_fp, label="fp")
# axs[1].plot(thresholds, pn_3_fn, label="fn")
# axs[1].legend()
# axs[1].set_title("pn3")
#
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(thresholds, accuracy_pn1, label="Leaf", color="blue")
ax.plot(thresholds, accuracy_pn3, label="Stem", color="orange")
ax.plot(thresholds, accuracy_mean, label="Combined", color="green")
ax.hlines(np.max(accuracy_pn3), thresholds[0], thresholds[-1], color="black", linestyle="--")
plt.text(x=-9, y=np.max(accuracy_pn3) + 0.0, s=f'{np.max(accuracy_pn3):.4f}', color='black', verticalalignment='bottom')
ax.hlines(np.max(accuracy_pn1), thresholds[0], thresholds[-1], color="black", linestyle="--")
plt.text(x=-9, y=np.max(accuracy_pn1) + 0.0, s=f'{np.max(accuracy_pn1):.4f}', color='black', verticalalignment='bottom')
ax.hlines(np.max(accuracy_mean), thresholds[0], thresholds[-1], color="black", linestyle="--")
plt.text(x=-9, y=np.max(accuracy_mean) + 0.0, s=f'{np.max(accuracy_mean):.4f}', color='black', verticalalignment='bottom')
ax.set_ylim(0, 1)
ax.set_xlim(-10, 10)
ax.legend()
ax.set_title("Threshold model accuracy")

tikzplotlib.save("plots/2024_felix/threshold_model.tex")

plt.show()

