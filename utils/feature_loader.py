import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def load_tsfresh_feature(exp_name, sensor, clean=False):
    if type(exp_name) is str:
        exp_name = [exp_name]
    if type(sensor) is str:
        sensor = [sensor]

    # load the data for all experiments
    y_all = None
    x_all = None
    for exp in exp_name:
        y_sensor = None
        x_sensor = None
        for s in sensor:
            # TODO adjust path
            f = pd.read_csv(f"../results/features_tsfresh/{exp}_{s}_features.csv")
            f_stim = f[f.columns[:788]]
            f_no = f[f.columns[788:]]
            for name in f_no.columns:
                rename_dict = {name: s + name[:-2]}  # Dictionary mapping old column name to new column name
                f_no = f_no.rename(columns=rename_dict)
            for name in f_stim.columns:
                rename_dict = {name: s + name}
                f_stim = f_stim.rename(columns=rename_dict)

            if clean:
                # todo adjust for each experiment
                if exp == "Exp44_Ivy2" and s == "pn1":
                    f_stim.drop(index=7, inplace=True)
                    f_no.drop(index=7, inplace=True)
                if exp == "Exp44_Ivy2" and s == "pn3":
                    f_stim.drop(index=7, inplace=True)
                    f_no.drop(index=7, inplace=True)
                # for i in f_stim.columns:
                #     if f_stim[i].isna().any():
                #         print(f"stim nan: {i}")
                #         print(f_stim[i])
                stim_nan = f_stim.columns[f_stim.isna().any()]
                no_nan = f_no.columns[f_no.isna().any()]
                f_nan = stim_nan.append(no_nan)
                # print(f"stim nan: {stim_nan}, {len(stim_nan)}")
                # print(f"no nan: {no_nan}, {len(no_nan)}")
                # print(f"f nan: {f_nan}, {len(f_nan)}")
                # remove all nan columns
                f_stim.drop(columns=f_nan, inplace=True)
                f_no.drop(columns=f_nan, inplace=True)
                # f_stim.dropna(axis=1, inplace=True)
                # f_no.dropna(axis=1, inplace=True)
            y_stim = np.ones((f_stim.shape[0],))
            y_no = np.zeros((f_no.shape[0],))

            if y_sensor is None:
                y_sensor = np.concatenate((y_no, y_stim), axis=0)
                x_sensor = pd.concat([f_no, f_stim], axis=0)
                print(f"shape x sensor1: {x_sensor.shape}")
            else:
                #y_sensor = np.concatenate((y_sensor, np.concatenate((y_no, y_stim), axis=0)), axis=0)
                print(f"to concat shapes: {x_sensor.shape}, {pd.concat([f_no, f_stim], axis=0).shape}")
                x_sensor = pd.concat([x_sensor, pd.concat([f_no, f_stim], axis=0)], axis=1)
                print(f"shape x sensor2: {x_sensor.shape}")
        if y_all is None:
            y_all = y_sensor
            x_all = x_sensor
        else:
            y_all = np.concatenate((y_all, y_sensor), axis=0)
            x_all = pd.concat([x_all, x_sensor], axis=0)


    return x_all, y_all


x, y = load_tsfresh_feature("Exp44_Ivy2", ["pn1"], clean=True)

# shuffle stuff and split into train and validation set; make reproducible by setting random state
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=14)

