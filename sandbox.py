import platform
import os

import pandas as pd
from datetime import timedelta
import matplotlib
import matplotlib.pyplot as plt

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

# res = pd.read_csv("results/naml_history.csv")
# res.drop(columns=["Unnamed: 0", "time", "runtime", "exception"], inplace=True)
# print(res.head())


df = pd.read_csv("results/ozone_peak.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)
exp_names = ["Exp44_Ivy2", "Exp45_Ivy4", "Exp46_Ivy0", "Exp47_Ivy5"]

for exp_name in exp_names:
    print(f"{exp_name}: {df[exp_name].mean()} +- {df[exp_name].std()}")

print(f"all experiments: {df.mean().mean()} +- {df.std().mean()}")

exit(22)


exp_names = ["Exp44_Ivy2", "Exp45_Ivy4", "Exp46_Ivy0", "Exp47_Ivy5"]  # "Exp44_Ivy2",
plotting = False
# if external drive is mounted:
# DIR = f"/Volumes/Data/watchplant/gas_experiments/ozone_cut/{exp_name}"
# else
# DIR = f"data/gas_experiments/ozone_cut/{exp_name}"

ozone_peak_df = pd.DataFrame(columns=exp_names, index=range(24))

for exp_name in exp_names:
    print(f"experiment {exp_name}")
    dir_path = "/Volumes/Data/watchplant"  # "data" or "/Volumes/Data/watchplant"
    DIR = f"{dir_path}/gas_experiments/ozone_cut/{exp_name}"
    experiments = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    timestamps = pd.read_csv(f"{dir_path}/gas_experiments/ozone/{exp_name}/times.csv")


    for i, ts in enumerate(timestamps["times"]):
        print(f"experiment number {i}")
        # get timings
        stimulus_application_begin = pd.to_datetime(ts, format="%Y-%m-%d %H:%M:%S")
        stimulus_application_end = stimulus_application_begin + timedelta(minutes=10)  # 10 min stimulus

        # load data
        path = os.path.join(DIR, f"experiment_{i}.csv")
        df = pd.read_csv(path, index_col=["timestamp"], date_format="%Y-%m-%d %H:%M:%S.%f")

        # get ozone
        df_ozone = df[["O3_1", "O3_2"]]
        df_ozone = df_ozone.dropna()
        # cut data
        df_ozone_app = cut_data(df_ozone, stimulus_application_begin, stimulus_application_end)
        df_ozone_rec = exclude_data(df_ozone, stimulus_application_begin, stimulus_application_end)

        # print(f"ozone: {df_ozone_app.head()}")
        # print(f"shapes: {df_ozone.shape} = {df_ozone_app.shape} + {df_ozone_rec.shape}")
        # plt.scatter(df_ozone_app.index, df_ozone_app["O3_1"], label="O3_1 app")
        # plt.scatter(df_ozone_rec.index, df_ozone_rec["O3_1"], label="O3_1 rec")
        # plt.scatter(df_ozone_app.index, df_ozone_app["O3_2"], label="O3_2 app")
        # plt.scatter(df_ozone_rec.index, df_ozone_rec["O3_2"], label="O3_2 rec")
        # plt.legend()
        #
        # plt.show()

        # get ozone peak
        ozone_peak = (df_ozone_app["O3_1"].max() + df_ozone_app["O3_2"].max()) / 2

        ozone_peak_df[exp_name].iloc[i] = ozone_peak

print(ozone_peak_df)
ozone_peak_df.to_csv("results/ozone_peak.csv")









