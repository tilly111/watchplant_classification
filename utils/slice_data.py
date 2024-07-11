'''
This script slices the raw air pollution experiment data into 130 minute intervals (60 min before and 70 min after the
stimulus). The sliced data are merged together and saved as a csv file in a specified folder.
'''

import platform

import pandas as pd
from datetime import timedelta, datetime
import matplotlib
import matplotlib.pyplot as plt
from utils.helper import load_dic

# for interactive plots
if platform.system() == "Darwin":
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    plt.rcParams.update({'font.size': 22})
    # pd.set_option('display.max_rows', None)
elif platform.system() == "Linux":
    matplotlib.use('TkAgg')


exp_names = ["Exp44_Ivy2", "Exp45_Ivy4", "Exp46_Ivy0", "Exp47_Ivy5"]

for exp_name in exp_names:
    print(exp_name)
    timestamps = pd.read_csv(f"/Volumes/Data/watchplant/gas_experiments/ozone/{exp_name}/times.csv")

    pn1 = load_dic(f"/Volumes/Data/watchplant/gas_experiments/ozone/{exp_name}/PN/P1")
    pn3 = load_dic(f"/Volumes/Data/watchplant/gas_experiments/ozone/{exp_name}/PN/P3")
    MU = load_dic(f"/Volumes/Data/watchplant/gas_experiments/ozone/{exp_name}/MU", device="MU")
    ozone = load_dic(f"/Volumes/Data/watchplant/gas_experiments/ozone/{exp_name}/Ozone", device="MU")

    pn1.rename(columns={"differential_potential": "differential_potential_pn1", "filtered": "filtered_pn1"}, inplace=True)
    pn3.rename(columns={"differential_potential": "differential_potential_pn3", "filtered": "filtered_pn3"}, inplace=True)

    print("loading done...")
    # experiment_data = pd.merge(pn1, pn3, on="timestamp", how="outer", copy=False)
    # experiment_data = pd.merge(experiment_data, MU, on="timestamp", how="outer", copy=False)
    # experiment_data = pd.merge(experiment_data, ozone, on="timestamp", how="outer", copy=False)

    for i, ts in enumerate(timestamps["times"]):
        print(ts)
        experiment_begin = pd.to_datetime(ts, format="%Y-%m-%d %H:%M:%S") - timedelta(minutes=60)
        # 10 min stimulus + 60 min recovery
        experiment_end = pd.to_datetime(ts, format="%Y-%m-%d %H:%M:%S") + timedelta(minutes=70)

        # get only the experiment data
        tmp_data_pn1 = pn1[pn1["timestamp"] >= experiment_begin]
        tmp_data_pn1 = tmp_data_pn1[tmp_data_pn1["timestamp"] <= experiment_end]

        tmp_data_pn3 = pn3[pn3["timestamp"] >= experiment_begin]
        tmp_data_pn3 = tmp_data_pn3[tmp_data_pn3["timestamp"] <= experiment_end]

        tmp_data_MU = MU[MU["timestamp"] >= experiment_begin]
        tmp_data_MU = tmp_data_MU[tmp_data_MU["timestamp"] <= experiment_end]

        tmp_ozone = ozone[ozone["timestamp"] >= experiment_begin]
        tmp_ozone = tmp_ozone[tmp_ozone["timestamp"] <= experiment_end]

        experiment_data = pd.merge(tmp_data_pn1, tmp_data_pn3, on="timestamp", how="outer", copy=False)
        experiment_data = pd.merge(experiment_data, tmp_data_MU, on="timestamp", how="outer", copy=False)
        experiment_data = pd.merge(experiment_data, tmp_ozone, on="timestamp", how="outer", copy=False)
        experiment_data.sort_values(by='timestamp', inplace=True)

        # save csv file
        experiment_data.to_csv(f"/Volumes/Data/watchplant/gas_experiments/ozone_cut/{exp_name}/experiment_{i}.csv", index=False)
