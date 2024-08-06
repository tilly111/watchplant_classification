import platform
import os

import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib
import matplotlib.pyplot as plt
from utils.helper import load_experiment_excel, load_experiment


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

def exclude_data(df, begin, end):
    df = df.drop(df[((df.index >= begin) & (df.index <= end))].index)
    return df


ozone_threhsold = 100
setups = [1, 2, 3, 4]

for setup in setups:

    df = load_experiment(f"/Volumes/Data/watchplant/2024_Vic_M1/combined/cyb{setup}_o3.csv", date_format="%Y-%m-%d %H:%M:%S")

    df["RMS_CH1_std"] = df["RMS_CH1"].rolling(window=100, min_periods=1).std()
    df["RMS_CH2_std"] = df["RMS_CH2"].rolling(window=100, min_periods=1).std()

    total_samples = df["O3"].count()

    y_values = df["O3"].dropna()

    all_X_data = None
    all_y_data = None
    for i in range(total_samples):
        if y_values[i] > ozone_threhsold:
            all_y_data = np.append(all_y_data, 1)
            # print(f"High ozone {y_values.index[i]} has value {y_values[i]}")
        else:
            all_y_data = np.append(all_y_data, 0)
            # print(f"Low ozone {y_values.index[i]} has value {y_values[i]}")

        df_tmp = cut_data(df, y_values.index[i], y_values.index[i] + timedelta(minutes=30))

        df_tmp.drop(columns=["O3"], inplace=True)
        df_tmp.dropna(inplace=True)
        df_tmp.index = df_tmp.index + pd.to_timedelta(0, unit='ms')
        # print(df_tmp.shape)
        # resample to have always the same length; maybe 144 length?
        # new_index = np.linspace(0, len(df) - 1, 144)
        df_resampled = df_tmp.resample('12S').mean()  # todo find better interpolation method
        df_resampled = df_resampled.interpolate(method='linear').head(144)

        if df_resampled.shape[0] != 144:
            print(f"wrong shape: {df_resampled.shape} at {y_values.index[i]} in setup {setup}")
            if df_resampled.shape[0] == 143:
                print("fix by padding same value (1)")
                df_resampled.loc[df_resampled.index[142] + pd.to_timedelta(12, unit='s')] = df_resampled.iloc[-1]
            else:
                print("unfixable... skipping this one")
                continue

        # df_resampled = np.interp(new_index, np.arange(len(df)), df_tmp['RMS_CH1'])
        # print(df_resampled.shape)

        # if df_tmp.shape[0] <= 100:
        #     plt.plot(df_resampled.index, df_resampled["RMS_CH1"], label="resampled")
        #     plt.plot(df_tmp.index, df_tmp["RMS_CH1"], label="original")
        #     plt.legend()
        #     plt.show()


        tmp_values = df_resampled.values

        tmp_values = tmp_values[np.newaxis, :]

        # print(tmp_values.shape)
        all_X_data = np.concatenate((all_X_data, tmp_values), axis=0) if all_X_data is not None else tmp_values

    all_y_data = all_y_data[1:]
    print(all_y_data)
    print(all_X_data.shape)
    print(all_y_data.shape)

    print(f"prop of high ozone: {np.sum(all_y_data) / all_y_data.shape[0]}")

    np.save(f"/Volumes/Data/watchplant/2024_Vic_M1/preprocessed/X_data_ozone_threshold_{ozone_threhsold}_ppb_setup_{setup}.npy", all_X_data)
    np.save(f"/Volumes/Data/watchplant/2024_Vic_M1/preprocessed/y_data_ozone_threshold_{ozone_threhsold}_ppb_setup_{setup}.npy", all_y_data)

exit(1)



df["O3_interpolated"] = df["O3"].interpolate(method="linear")




all_samples = df["O3_interpolated"].count()
samples_larger_than_70 = df[df["O3_interpolated"] > 70]["O3_interpolated"].count()
samples_larger_than_60 = df[df["O3_interpolated"] > 60]["O3_interpolated"].count()
print(f"all samples: {all_samples}")
print(f"Percentage of samples larger than 70: {samples_larger_than_70 / all_samples * 100}%")
print(f"Percentage of samples larger than 60: {samples_larger_than_60 / all_samples * 100}%")

plt.figure()
plt.hist(df["O3_interpolated"], bins=100)
plt.axvline(70, color='r', linestyle='dashed', linewidth=1, label='70')
plt.axvline(60, color='g', linestyle='dashed', linewidth=1, label='60')
plt.show()

print(df.head())









