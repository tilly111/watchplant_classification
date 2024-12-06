import platform
import os

import pandas as pd
import random
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

stim = pd.read_csv("/Volumes/Data/watchplant/gas_experiments/cut_samples/stim.csv", header=None)
no = pd.read_csv("/Volumes/Data/watchplant/gas_experiments/cut_samples/no.csv", header=None)


idxs = random.sample(range(76), 53)
stim_train = stim.iloc[idxs].reset_index(drop=True)
no_train = no.iloc[idxs].reset_index(drop=True)

stim_test = stim.drop(idxs).reset_index(drop=True)
no_test = no.drop(idxs).reset_index(drop=True)



train_class = np.concatenate([np.zeros((no_train.shape[0],)), np.ones((stim_train.shape[0],))])
test_class = np.concatenate([np.zeros((no_test.shape[0],)), np.ones((stim_test.shape[0],))])
train_class = pd.DataFrame(train_class, columns=["class"])
test_class = pd.DataFrame(test_class, columns=["class"])

train_save = pd.concat([no_train, stim_train], axis=0, ignore_index=True)
test_save = pd.concat([no_test, stim_test], axis=0)

print(train_class.shape)
print(test_class.shape)
print(train_save.shape)
print(test_save.shape)

train_class.to_csv("/Volumes/Data/watchplant/gas_experiments/cut_samples/y_train.csv", index=False)
test_class.to_csv("/Volumes/Data/watchplant/gas_experiments/cut_samples/y_test.csv", index=False)
train_save.to_csv("/Volumes/Data/watchplant/gas_experiments/cut_samples/X_train.csv", index=False)
test_save.to_csv("/Volumes/Data/watchplant/gas_experiments/cut_samples/X_test.csv", index=False)










