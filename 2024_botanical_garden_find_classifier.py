import platform
from tqdm import tqdm
import os

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import naiveautoml

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from utils.feature_loader import load_botanical

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from sklearn.metrics import f1_score, make_scorer
import logging


def calc_best_classifier(stimuli, channel, path_to_data):
    scoring = make_scorer(f1_score, average='macro')
    naml = naiveautoml.NaiveAutoML(max_hpo_iterations=1024, show_progress=True, scoring=scoring,
                                   max_hpo_iterations_without_imp=100, num_cpus=1, kwargs_as={'excluded_components': {"learner": ["HistGradientBoostingClassifier"]}})  # , kwargs_as={'excluded_components': {"learner": ["HistGradientBoostingClassifier"]}}

    x, y = load_botanical(stimuli, channel, path=path_to_data)


    # X_analysis, _, y_analysis, _ = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    X_analysis, _, y_analysis, _ = train_test_split(x, y, test_size=0.2, random_state=42)
    y_analysis.loc[y_analysis["class"] == stimuli[0]] = 0
    y_analysis.loc[y_analysis["class"] == stimuli[1]] = 1
    y_analysis = y_analysis.astype(int)

    # TODO SMOTE?
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_analysis, y_analysis = smote.fit_resample(X_analysis, y_analysis)
    # print(f"[upsampled] X_train: {X_train.shape}, y_train: {y_train.shape}")

    y_analysis = y_analysis["class"].to_numpy().ravel()  # .to_numpy().ravel()

    naml.fit(X_analysis, y_analysis)

    # print("---------------------------------")
    # print(naml.chosen_model)
    # print("---------------------------------")
    # print(naml.history)

    naml.history.to_csv(
            f"results/2024_botanical_garden/autoML_classifiers/naml_history_{'_'.join(channel)}tw_60_{'_'.join(stimuli)}_with_smote.csv")



if __name__ == "__main__":
    # for interactive plots
    if platform.system() == "Darwin":
        matplotlib.use('QtAgg')
        pd.set_option('display.max_columns', None)
        plt.rcParams.update({'font.size': 20})
        pd.set_option('display.max_rows', None)
        path_to_data = "/Volumes/Data/watchplant/"
    elif platform.system() == "Linux":
        # matplotlib.use('TkAgg')  # TODO removed because of server
        path_to_data = "data_preprocessed/"


    # do logging
    # logger = logging.getLogger('naiveautoml')
    # logger.setLevel(logging.INFO)
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # ch.setFormatter(formatter)
    # logger.addHandler(ch)

    # cold, day, dry, night, rain, warm, wind, windless
    stimuli_settings = [["cold", "warm"],
                        ["day", "night"],
                        ["dry", "rain"],
                        ["wind", "windless"]]

    pbar = tqdm(total=len(stimuli_settings) * 2)
    futures = []
    # for stimuli in stimuli_settings:
    #         for channel in [["CH1"], ["CH2"]]:
    #             calc_best_classifier(stimuli, channel, path_to_data)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for stimuli in stimuli_settings:
            for channel in [["CH1"], ["CH2"]]:
                executor.submit(calc_best_classifier, stimuli, channel, path_to_data)
        # Attach the callback to each future
        def _cb(future):
            pbar.update(1)