import pandas as pd
import numpy as np

import constants
from sklearn.model_selection import train_test_split


def load_tsfresh_feature(exp_name, sensor, clean=False, split=False):
    '''
    Load the tsfresh features for the given experiments and sensors
    :param exp_name: experiments you want to load
    :param sensor: sensors you want to load
    :param clean: is ignored when split is true, otherwise remove all nan columns or wrong experiments
    :param split: returns only the train data if true!
    :return:
    '''
    if type(exp_name) is str:
        exp_name = [exp_name]
    if type(sensor) is str:
        sensor = [sensor]

    if split:
        X_train = pd.read_csv(f"data_preprocessed/split_data/X_train_{sensor}.csv")
        y_train = pd.read_csv(f"data_preprocessed/split_data/y_train_{sensor}.csv")
        return X_train, y_train

    # load the data for all experiments
    y_all = None
    x_all = None
    for exp in exp_name:
        y_sensor = None
        x_sensor = None
        for s in sensor:
            print(f"loading {exp} {s}")
            # TODO adjust path
            f = pd.read_csv(f"data_preprocessed/features_tsfresh/{exp}_{s}_features.csv")
            f_stim = f[f.columns[:788]]
            f_no = f[f.columns[788:]]
            for name in f_no.columns:
                rename_dict = {name: s + name[:-2]}  # Dictionary mapping old column name to new column name
                # rename_dict = {name: (s + name[:-2]).replace("pn3", "pn1")}  # TODO if you want to have pn1 after pn3
                f_no = f_no.rename(columns=rename_dict)
            for name in f_stim.columns:
                rename_dict = {name: s + name}
                # rename_dict = {name: (s + name).replace("pn3", "pn1")}
                f_stim = f_stim.rename(columns=rename_dict)

            if clean:
                # todo adjust for each experiment
                if exp == "Exp44_Ivy2" and s == "pn1":
                    f_stim.drop(index=7, inplace=True)
                    f_no.drop(index=7, inplace=True)
                if exp == "Exp44_Ivy2" and s == "pn3":
                    f_stim.drop(index=7, inplace=True)
                    f_no.drop(index=7, inplace=True)
                if exp == "Exp45_Ivy4" and s == "pn3":
                    f_stim.drop(index=0, inplace=True)
                    f_no.drop(index=0, inplace=True)
                # for i in f_stim.columns:
                #     if f_stim[i].isna().any():
                #         print(f"stim nan: {i}")
                #         print(f_stim[i])
                stim_nan = f_stim.columns[f_stim.isna().any()]
                no_nan = f_no.columns[f_no.isna().any()]
                f_nan = stim_nan.append(no_nan)
                # print(f"f_nan: {f_nan}")
                # print(f"stim nan: {stim_nan}, {len(stim_nan)}")
                # print(f"no nan: {no_nan}, {len(no_nan)}")
                # print(f"f nan: {f_nan}, {len(f_nan)}")
                # remove all nan columns
                f_stim.drop(columns=f_nan, inplace=True)
                f_no.drop(columns=f_nan, inplace=True)
                # f_stim.dropna(axis=1, inplace=True)
                # f_no.dropna(axis=1, inplace=True)
            y_stim = pd.DataFrame(np.ones((f_stim.shape[0],)), columns=["y"])
            y_no = pd.DataFrame(np.zeros((f_no.shape[0],)), columns=["y"])
            tmp_x_sensors = pd.concat([f_no, f_stim], axis=0).reset_index(drop=True)
            tmp_y_sensors = pd.concat([y_no, y_stim], axis=0).reset_index(drop=True)
            print(f"shapes of the individual sensors: {tmp_x_sensors.shape}, {tmp_y_sensors.shape}")
            if y_sensor is None:
                x_sensor = tmp_x_sensors
                y_sensor = tmp_y_sensors
            else:
                if exp == "Exp45_Ivy4" and s == "pn3" and len(sensor) > 1:
                    # tmp_x_sensors.reset_index(drop=True, inplace=True)
                    tmp_x_sensors.index = [30, 31, 32, 33, 34, 35, 36, 37]
                    tmp_x_sensors = pd.concat([pd.DataFrame(index=range(30)), tmp_x_sensors], axis=0)
                x_sensor = pd.concat([x_sensor, tmp_x_sensors], axis=1)
                print(f"shape x after concat: {x_sensor.shape}")

        if y_all is None:
            y_all = y_sensor
            x_all = x_sensor
        else:
            y_all = pd.concat([y_all, y_sensor], axis=0)
            x_all = pd.concat([x_all, x_sensor], axis=0)

    # replace all " with _ to dont run into parsing errors later on
    x_all.columns = [col.replace('"', '_') for col in x_all.columns]

    return x_all, y_all


def load_eddy_feature(exp_name, sensors, feature_selection=None, manual_features_stim=None, manual_features_no=None):
    features = None
    all_feature_names = None
    for sensor in sensors:
        features_tmp = pd.read_csv(f"data_preprocessed/features_median_filter/{exp_name}_{sensor}_features.csv")
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

    return x_train, y_train, all_feature_names

# x, y = load_tsfresh_feature("Exp44_Ivy2", ["pn1"], clean=True)
#
# # shuffle stuff and split into train and validation set; make reproducible by setting random state
# x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=14)
#
