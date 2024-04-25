import pandas as pd
import glob
import os


def cut_data(df, begin, end):
    try:
        df = df[df["timestamp"] >= begin]
        df = df[df["timestamp"] <= end]
    except KeyError:
        df = df[df.index >= begin]
        df = df[df.index <= end]

    return df


def load_dic(dic_path, device="PN"):
    all_files = glob.glob(os.path.join(dic_path, "*.csv"))

    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    if device == "PN":
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S:%f")
        df["timestamp"] = df["timestamp"].astype('datetime64[ms]')
    elif device == "MU":
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
        df["timestamp"] = df["timestamp"].astype('datetime64[ms]')
        # df["timestamp"] = df["timestamp"].apply(lambda x: x + timedelta(milliseconds=1))
    df.sort_values(by='timestamp', inplace=True)
    return df

def load_experiment(data_path):
    df = pd.read_csv(data_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S.%f")
    df["timestamp"] = df["timestamp"].astype('datetime64[ms]')
    df.sort_values(by='timestamp', inplace=True)
    df["timestamp"] = df["timestamp"] - df["timestamp"].iloc[0]
    df.set_index('timestamp', inplace=True)

    return df
