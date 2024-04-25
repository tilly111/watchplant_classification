import platform
import matplotlib
import matplotlib.pyplot as plt
from datetime import timedelta
import pandas as pd
import numpy as np
import emd

from utils.helper import *

# if platform.system() == "Darwin":
#     matplotlib.use('QtAgg')
#     pd.set_option('display.max_columns', None)
#     plt.rcParams.update({'font.size': 22})
#     # pd.set_option('display.max_rows', None)
# elif platform.system() == "Linux":
#     matplotlib.use('TkAgg')


def emd_filter(ts: pd.DataFrame, c: int = 6, verbose: bool = True) -> pd.DataFrame:
    tmp_dict = {}
    for i, channel in enumerate(ts.keys()):
        time_d = (ts.index[-1] - ts.index[0]).total_seconds()
        ts_tmp = ts.to_numpy()[:, i]
        sample_rate = ts_tmp.shape[0] / (time_d)  # should be Hz
        if verbose:
            print(f"time delta: {time_d} should be roughly 7800?; individual samples: {ts_tmp.shape[0]}")
            print(f"EMD filter detected sample rate: {sample_rate} Hz")
        if sample_rate > 100:
            c = 6
        elif sample_rate < 5:
            c = 1
        elif sample_rate < 50:
            c = 6
        else:
            c = 6

        imf = emd.sift.sift(ts_tmp)

        reconstructed_signal = np.zeros((imf.shape[0], 1))
        for j in range(imf.shape[1]):
            if j < c:  # skipping the first c IMF ... 6 or 7 seems to be good
                continue
            reconstructed_signal[:, 0] += imf[:, j]

        tmp_dict[channel] = reconstructed_signal.squeeze()
    # put back to dataframe
    denoised_signal = pd.DataFrame(tmp_dict)
    denoised_signal.index = ts.index

    return denoised_signal


# load data set
# exp_names = ["Exp47_Ivy5"]
# col = ["differential_potential_pn1", "differential_potential_pn3"]
# time_d = 12  # in minutes
#
#
# for exp_name in exp_names:
#     timestamps = pd.read_csv(f"/Volumes/Data/watchplant/gas_experiments/ozone/{exp_name}/times.csv")
#     for i in range(17):
#         experiment_start = pd.to_datetime(timestamps["times"][i], format="%Y-%m-%d %H:%M:%S")
#         experiment_end = experiment_start + timedelta(minutes=10)
#         file_name = f"/Volumes/Data/watchplant/gas_experiments/ozone_cut/{exp_name}/experiment_{i}.csv"
#         ts = pd.read_csv(file_name, usecols=["timestamp"] + col, index_col=["timestamp"], date_format="%Y-%m-%d %H:%M:%S.%f")
#         begin = experiment_start - timedelta(minutes=1)
#         end = experiment_end + timedelta(minutes=1)
#         print(f"begin: {begin}, end: {end}")
#         print(ts.shape)
#         ts = cut_data(ts, begin, end)
#         print(ts.shape)
#         ts.interpolate(method="linear", inplace=True)
#         Vref = 2.5
#         Gain = 4
#         databits = 8388608
#         ts["differential_potential_pn1"] = ((ts["differential_potential_pn1"] / databits) - 1) * Vref / Gain * 1000
#         ts["differential_potential_pn3"] = ((ts["differential_potential_pn3"] / databits) - 1) * Vref / Gain * 1000
#
#         if ts.iloc[0].isna().any():
#             print("NANs in the beginning")
#             ts = ts.dropna()
#
#         d_sig = emd_filtering(ts, c=6, verbose=True)
#
#         # ts.plot()
#         plt.plot(ts["differential_potential_pn1"], label="original ch1")
#         plt.plot(ts["differential_potential_pn3"], label="original ch2")
#         plt.axvspan(experiment_start, experiment_end, facecolor='0.2', alpha=0.5)
#         plt.plot(d_sig["differential_potential_pn1"], label="denoised ch1")
#         plt.plot(d_sig["differential_potential_pn3"], label="denoised ch2")
#         plt.legend()
#         # d_sig.plot()
#         plt.show()
#         # ts = ts.to_numpy()
#         # print(f"ts shape: {ts.shape}")
#         # sample_rate = ts.shape[0] / (60*time_d)  # should be Hz
#         # print(f"sample rate: {sample_rate} Hz")
#         #
#         # imf = emd.sift.sift(ts)
#         # IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'hilbert')
#         # # freq_range = (0.1, 1200, 600, 'log')
#         # # f, hht = emd.spectra.hilberthuang(IF, IA, freq_range, sum_time=False)
#         #
#         # emd.plotting.plot_imfs(imf)
#         # # fig = plt.figure(figsize=(10, 6))
#         # # emd.plotting.plot_hilberthuang(hht, ts["timestamp"], f,
#         # #                                time_lims=(2, 4), freq_lims=(0.1, 15),
#         # #                                fig=fig, log_y=True)
#         # # ts.plot()
#         # print("IMFs: ", imf.shape)
#         # reconstructed_signal = np.zeros((imf.shape[0], 1))
#         # print(f"rec sig: {reconstructed_signal.shape}")
#         # for j in range(imf.shape[1]):
#         #     if j < 7:  # skipping the first IMF ... 6 or 7 seems to be good
#         #         continue
#         #     reconstructed_signal[:, 0] += imf[:, j]
#         #     print(f"rec sig: {reconstructed_signal.shape}")
#         # print("reconstructed shape: ", reconstructed_signal.shape)
#         # plt.figure()
#         # plt.plot(reconstructed_signal)
#         #
#         # # put back to dataframe
#         # denoised_signal = pd.DataFrame({"differential_potential_pn1": reconstructed_signal[0, :]})
#         # denoised_signal.index = ts.index
#         #
#         # denoised_signal.plot()
#         # plt.show()
#         # # todo return signal
#
# # do emd denoising