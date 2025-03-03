import platform
import os

import pandas as pd
import random
import numpy as np
from datetime import timedelta
import matplotlib
import matplotlib.pyplot as plt
from utils.helper import load_experiment_excel, load_experiment
from sklearn.metrics import classification_report


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

# Example data
y_true = [0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 1]

# Generate classification report as a dictionary
report = classification_report(y_true, y_pred, output_dict=True)

# Convert to DataFrame
report_df = pd.DataFrame(report).transpose()

# Save to CSV
report_df.to_csv('results/2024_botanical_garden/report/classification_report.csv', index=True)









