import platform
import glob
import os
import constants

import pandas as pd
from datetime import timedelta, datetime
import matplotlib
import matplotlib.pyplot as plt
from preprocessing.emd_filter import emd_filter
from utils.helper import load_experiment

from features import calc_all_features, calc_ts_fresh

# for interactive plots
if platform.system() == "Darwin":
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    plt.rcParams.update({'font.size': 22})
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
elif platform.system() == "Linux":
    matplotlib.use('TkAgg')


def cut_data(df, begin, end):
    df = df.loc[df.index >= begin]
    df = df.loc[df.index <= end]
    return df

# res = pd.read_csv("results/naml_history.csv")
# res.drop(columns=["Unnamed: 0", "time", "runtime", "exception"], inplace=True)
# print(res.head())


# exp_names = ["Exp44_Ivy2"]  # "Exp44_Ivy2",, "Exp45_Ivy4", "Exp46_Ivy0", "Exp47_Ivy5"
# plotting = False
# if external drive is mounted:
# DIR = f"/Volumes/Data/watchplant/gas_experiments/ozone_cut/{exp_name}"
# else
# DIR = f"data/gas_experiments/ozone_cut/{exp_name}"

# for exp_name in exp_names:
#     for i in range(17):
#         dir_path = f"data/gas_experiments/ozone_cut/{exp_name}/experiment_{i}.csv"
#         df = load_experiment(dir_path)
#         print(df.head())  # "differential_potential_CH1"
#         df_ch1 = df.dropna(subset=["differential_potential_CH1"])
#         df_ch2 = df.dropna(subset=["differential_potential_CH2"])
#         # df_ch1.plot(y="differential_potential_CH1")
#         # df_ch2.plot(y="differential_potential_CH2")
#         df_ch1['differential_potential_CH1'].hist(bins=100, edgecolor='black')
#         plt.show()

    # dir_path = "data"  # or "/Volumes/Data/watchplant"
    # DIR = f"{dir_path}/gas_experiments/ozone_cut/{exp_name}"
    # experiments = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    # timestamps = pd.read_csv(f"{dir_path}/gas_experiments/ozone/{exp_name}/times.csv")
    #
    # features_pn1 = None  # pd.DataFrame(columns=constants.ALL_FEATURES)
    # features_pn3 = None  # pd.DataFrame(columns=constants.ALL_FEATURES)
    # features_mu_ch1 = None  # pd.DataFrame(columns=constants.ALL_FEATURES)
    # features_mu_ch2 = None  # pd.DataFrame(columns=constants.ALL_FEATURES)
    # for i, ts in enumerate(timestamps["times"]):
    #     print(f"experiment number {i}")
    #     # get timings
    #     stimulus_application_begin = pd.to_datetime(ts, format="%Y-%m-%d %H:%M:%S")
    #     stimulus_application_end = stimulus_application_begin + timedelta(minutes=10)  # 10 min stimulus
    #
    #     # load data
    #     path = os.path.join(DIR, f"experiment_{i}.csv")
    #     df = pd.read_csv(path, index_col=["timestamp"], date_format="%Y-%m-%d %H:%M:%S.%f")
    #
    #     # get ozone
    #     df_ozone = df[["O3_1", "O3_2"]]
    #     df_ozone = df_ozone.dropna()
    #
    #     # drop unnecessary values
    #     df_pn_1 = df[["differential_potential_pn1"]]
    #     df_pn_3 = df[["differential_potential_pn3"]]
    #     df_pn_1 = df_pn_1.dropna()
    #     df_pn_3 = df_pn_3.dropna()
    #     if df_pn_1.empty:
    #         print(f"pn 1 empty in experiment {i}")
    #     if df_pn_3.empty:
    #         print(f"pn 3 empty in experiment {i}")
    #     df_mu = df[["differential_potential_CH1", "differential_potential_CH2"]]
    #     df_mu = df_mu.dropna()
    #
    #     # scaling
    #     BPO_SCALING = 0.001  # impedance scaling that we have mv
    #     BPO_OFFSET = 512000  # here we have no offset
    #     df_mu["differential_potential_CH1"] = (df_mu["differential_potential_CH1"] - BPO_OFFSET) * BPO_SCALING
    #     df_mu["differential_potential_CH2"] = (df_mu["differential_potential_CH2"] - BPO_OFFSET) * BPO_SCALING
    #
    #     Vref = 2.5
    #     Gain = 4
    #     databits = 8388608
    #     df_pn_1["differential_potential_pn1"] = ((df_pn_1[
    #                                                   "differential_potential_pn1"] / databits) - 1) * Vref / Gain * 1000
    #     df_pn_3["differential_potential_pn3"] = ((df_pn_3[
    #                                                   "differential_potential_pn3"] / databits) - 1) * Vref / Gain * 1000
    #
    #     # some static filter
    #     df_pn_1.drop(df_pn_1[df_pn_1["differential_potential_pn1"] >= 200].index, inplace=True)
    #     df_pn_1.drop(df_pn_1[df_pn_1["differential_potential_pn1"] <= -200].index, inplace=True)
    #     df_pn_3.drop(df_pn_3[df_pn_3["differential_potential_pn3"] >= 200].index, inplace=True)
    #     df_pn_3.drop(df_pn_3[df_pn_3["differential_potential_pn3"] <= -200].index, inplace=True)
    #
    #     # filter stuff
    #     N = 10
    #     df_pn_1["differential_potential_pn1"] = df_pn_1["differential_potential_pn1"].rolling(window=N).median()
    #     df_pn_3["differential_potential_pn3"] = df_pn_3["differential_potential_pn3"].rolling(window=N).median()
    #     df_mu["differential_potential_CH1"] = df_mu["differential_potential_CH1"].rolling(
    #         window=int(N / 2)).median()
    #     df_mu["differential_potential_CH2"] = df_mu["differential_potential_CH2"].rolling(
    #         window=int(N / 2)).median()
    #
    #     # downsample the signal
    #     df_pn_1 = df_pn_1.resample('0.5S').mean()
    #     df_pn_3 = df_pn_3.resample('0.5S').mean()
    #
    #     # cut data into time slices
    #     bg_start = stimulus_application_begin - timedelta(minutes=20)
    #     bg_end = stimulus_application_begin - timedelta(minutes=10)
    #     no_start = stimulus_application_begin - timedelta(minutes=10)
    #     no_end = stimulus_application_begin
    #
    #     if not df_pn_1.empty:
    #         bg_pn_1 = cut_data(df_pn_1, bg_start, bg_end)
    #         no_pn_1 = cut_data(df_pn_1, no_start, no_end)
    #         stim_pn_1 = cut_data(df_pn_1, stimulus_application_begin, stimulus_application_end)
    #
    #     if not df_pn_3.empty:
    #         bg_pn_3 = cut_data(df_pn_3, bg_start, bg_end)
    #         no_pn_3 = cut_data(df_pn_3, no_start, no_end)
    #         stim_pn_3 = cut_data(df_pn_3, stimulus_application_begin, stimulus_application_end)
    #
    #     bg_mu = cut_data(df_mu, bg_start, bg_end)
    #     no_mu = cut_data(df_mu, no_start, no_end)
    #     stim_mu = cut_data(df_mu, stimulus_application_begin, stimulus_application_end)
    #
    #     # calculate features
    #     if not df_pn_1.empty:
    #         if features_pn1 is None:
    #             features_pn1 = calc_ts_fresh(bg_pn_1["differential_potential_pn1"],
    #                                                 no_pn_1["differential_potential_pn1"],
    #                                                 stim_pn_1["differential_potential_pn1"])
    #         else:
    #             tmp = calc_ts_fresh(bg_pn_1["differential_potential_pn1"],
    #                                                 no_pn_1["differential_potential_pn1"],
    #                                                 stim_pn_1["differential_potential_pn1"])
    #             features_pn1 = pd.concat([features_pn1, tmp], axis=0)
    #
    #     if not df_pn_3.empty:
    #         if features_pn3 is None:
    #             features_pn3 = calc_ts_fresh(bg_pn_3["differential_potential_pn3"],
    #                                                 no_pn_3["differential_potential_pn3"],
    #                                                 stim_pn_3["differential_potential_pn3"])
    #         else:
    #             tmp = calc_ts_fresh(bg_pn_3["differential_potential_pn3"],
    #                                                     no_pn_3["differential_potential_pn3"],
    #                                                     stim_pn_3["differential_potential_pn3"])
    #             features_pn3 = pd.concat([features_pn3, tmp], axis=0)
    #
    #     if features_mu_ch1 is None:
    #         features_mu_ch1 = calc_ts_fresh(bg_mu["differential_potential_CH1"],
    #                                             no_mu["differential_potential_CH1"],
    #                                             stim_mu["differential_potential_CH1"])
    #     else:
    #         tmp = calc_ts_fresh(bg_mu["differential_potential_CH1"],
    #                                                    no_mu["differential_potential_CH1"],
    #                                                    stim_mu["differential_potential_CH1"])
    #         features_mu_ch1 = pd.concat([features_mu_ch1, tmp], axis=0)
    #
    #     if features_mu_ch2 is None:
    #         features_mu_ch2 = calc_ts_fresh(bg_mu["differential_potential_CH2"],
    #                                             no_mu["differential_potential_CH2"],
    #                                             stim_mu["differential_potential_CH2"])
    #     else:
    #         tmp = calc_ts_fresh(bg_mu["differential_potential_CH2"],
    #                                                    no_mu["differential_potential_CH2"],
    #                                                    stim_mu["differential_potential_CH2"])
    #         features_mu_ch2 = pd.concat([features_mu_ch2, tmp], axis=0)
    #
    # if not df_pn_1.empty:
    #     features_pn1.to_csv(f"results/features_tsfresh/{exp_name}_pn1_features.csv", index=False)
    # if not df_pn_3.empty:
    #     features_pn3.to_csv(f"results/features_tsfresh/{exp_name}_pn3_features.csv", index=False)
    # features_mu_ch1.to_csv(f"results/features_tsfresh/{exp_name}_mu_ch1_features.csv", index=False)
    # features_mu_ch2.to_csv(f"results/features_tsfresh/{exp_name}_mu_ch2_features.csv", index=False)
