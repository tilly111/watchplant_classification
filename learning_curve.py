import platform
import sys

import constants
import os
import json
from tqdm import tqdm

import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import matplotlib
import matplotlib.pyplot as plt
from itertools import compress

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing._data import Normalizer
from sklearn.feature_selection._univariate_selection import GenericUnivariateSelect
from sklearn.feature_selection import RFECV, SequentialFeatureSelector, RFE, VarianceThreshold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import auc, get_scorer
from sklearn.base import clone

from utils.learner_pipeline import get_pipeline_for_features

from utils.feature_loader import load_tsfresh_feature, load_eddy_feature

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# for interactive plots
if platform.system() == "Darwin":
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    plt.rcParams.update({'font.size': 20})
    pd.set_option('display.max_rows', None)
elif platform.system() == "Linux":
    # matplotlib.use('TkAgg')  # TODO removed because of server
    pass
elif platform.system() == "Windows":
    # TODO
    pass


def get_learning_curve(learner, X, y, seed, schedule, scoring="roc_auc"):
    auc_train = []
    auc_val = []
    for i, a in enumerate(schedule):
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=a, stratify=y, random_state=seed)
        l_copy = clone(learner)
        l_copy.fit(X_train, y_train.values.ravel())
        if scoring == "roc_auc":
            auc_train.append(roc_auc_score(y_train, l_copy.predict_proba(X_train)[:, 1]))
            auc_val.append(roc_auc_score(y_val, l_copy.predict_proba(X_val)[:, 1]))
        elif scoring == "accuracy":
            auc_train.append(accuracy_score(y_train, l_copy.predict(X_train)))
            auc_val.append(accuracy_score(y_val, l_copy.predict(X_val)))
    return auc_train, auc_val


def get_learning_curves(learner, X, y, repeats, first_anchor=0.1, last_anchor=0.8, steps=10, filename=None, scoring="roc_auc"):
    schedule = np.linspace(first_anchor, last_anchor, steps)

    if filename is None or not os.path.isfile(filename):
        lcs = None
    else:
        lcs = pd.read_csv(filename)

    pbar = tqdm(total=repeats)
    futures = []

    with ProcessPoolExecutor(max_workers=6) as executor:
        for seed in range(repeats):
            if lcs is None or len(lcs) < seed:
                futures.append(
                    (seed,
                     executor.submit(
                         get_learning_curve,
                         learner,
                         X,
                         y,
                         seed,
                         schedule,
                         scoring
                     )
                     )
                )
            else:
                pbar.update(1)

        # Attach the callback to each future
        def _cb(future):
            pbar.update(1)

        for seed, future in futures:
            future.add_done_callback(_cb)

        # await results
        as_completed([f for k, f in futures])

    # close pbar
    pbar.close()

    # store results
    if futures:
        lcs_new = pd.DataFrame([future.result()[1] for seed, future in futures], columns=schedule)
        lcs = lcs_new if lcs is None else pd.concat([lcs, lcs_new], axis=0)
    lcs.to_csv(filename, index=False)

    return lcs


def get_score_for_features(classifier, X, y, feature_list, repeats):
    X_red = X[feature_list]
    # pl_interpretable = get_pipeline_for_features(classifier, X, y, feature_list)
    pl_interpretable = classifier
    scorer = get_scorer("roc_auc")

    results = []
    for seed in range(repeats):
        X_train, X_val, y_train, y_val = train_test_split(X_red, y, stratify=y, train_size=0.8, random_state=seed)
        l = clone(pl_interpretable).fit(X_train, y_train.values.ravel())
        results.append(scorer(l, X_val, y_val))
    return results


def get_scores_for_feature_combinations_based_on_previous_selections(classifier, X, y, repeats_per_size, df_last_stage,
                                                                     num_combos_from_last_stage):
    if df_last_stage is None:
        combos_for_k = [[c] for c in X.columns]
    else:
        accepted_combos_last_stage = df_last_stage["combo"][:num_combos_from_last_stage]
        combos_for_k = []
        for accepted_prev_combo in accepted_combos_last_stage:
            for c in X.columns:
                if c not in accepted_prev_combo:  # and c > accepted_prev_combo[-1]:
                    new_combo = sorted(accepted_prev_combo + [c])
                    if new_combo not in combos_for_k:
                        combos_for_k.append(new_combo)

    pbar = tqdm(total=len(combos_for_k))
    rows = []

    # Using ThreadPoolExecutor to parallelize the function calls
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:

        # Submit tasks to the executor
        futures = [
            executor.submit(get_score_for_features, classifier, X, y, combo, repeats_per_size[len(combo)])
            for combo in combos_for_k
        ]

        # Attach the callback to each future
        def _cb(future):
            pbar.update(1)

        for future in futures:
            future.add_done_callback(_cb)

        # await results
        as_completed(futures)

        # Process results as they complete
        for combo, future in zip(combos_for_k, futures):
            scores_for_combo = future.result()
            rows.append([combo, scores_for_combo, np.mean(scores_for_combo)])

    pbar.close()

    return pd.DataFrame(rows, columns=["combo", "scores", "score_mean"]).sort_values("score_mean", ascending=False)


def get_scores_for_feature_combinations(classifier, X, y, max_size, dir, repeats_per_size, num_combos_from_last_stage,
                                        sensors_names="pn1"):
    dfs = {}

    for k in range(1, max_size + 1):

        path = f"{dir}results/feature_combinations/{sensors_names}_feature_selection_results_{k}.csv"
        # path = f"{dir}results/feature_combinations/feature_selection_results_{k}.csv"
        if os.path.isfile(path):
            print(f"Loading {path}...")
            dfs[k] = pd.read_csv(path)
            dfs[k]["combo"] = [json.loads(e.replace("'", '"')) for e in dfs[k]["combo"]]
            dfs[k]["scores"] = [json.loads(e) for e in dfs[k]["scores"]]
        else:

            combos_from_last_stage = None if k == 1 else [set()]
            if k == 1:
                df_for_last_k = None
            if k > 1:
                df_for_last_k = dfs[k - 1]  # .drop(columns=attributes_excluded_in_multivar_importance)
            dfs[k] = get_scores_for_feature_combinations_based_on_previous_selections(classifier, X, y,
                                                                                      repeats_per_size, df_for_last_k,
                                                                                      num_combos_from_last_stage[
                                                                                          k] if k > 1 else 0)
        dfs[k].to_csv(path, index=False)
    return dfs


if __name__ == '__main__':
    # local machine with hard drive mounted
    if platform.system() == "Darwin":
        dir = "/Volumes/Data/watchplant/Results/2024_felix/"
        plotting = True
    else:  # on server
        dir = "/abyss/home/code/watchplant_classification/"
        plotting = False

    exp_names = ["Exp44_Ivy2", "Exp45_Ivy4", "Exp46_Ivy0", "Exp47_Ivy5"]

    sensors = []
    for arg in sys.argv:
        if arg in constants.SENSOR_NAMES:
            sensors.append(str(arg))
    sensors_names = "_".join(sensors)

    print(f"Utilizing {os.cpu_count()} cpus.")
    print(f"Using sensors: {sensors}")

    if len(sensors) == 1 and sensors[0] == "pn1":  # based on roc auc score
        data_pre_processor = None
        learner = RandomForestClassifier(criterion='entropy', max_features=7,
                                         min_samples_leaf=6,
                                         min_samples_split=14, n_estimators=512,
                                         warm_start=True)
    elif len(sensors) == 1 and sensors[0] == "pn3":  # based on roc auc score
        data_pre_processor = Normalizer()
        learner = ExtraTreesClassifier(bootstrap=True, criterion='entropy',
                                       max_features=0.7074865514350775,
                                       min_samples_leaf=2, min_samples_split=4,
                                       n_estimators=512, warm_start=True)
    elif len(sensors) == 2 and sensors[0] == "pn1" and sensors[1] == "pn3":  # based on roc auc score
        # data_pre_processor = GenericUnivariateSelect(mode='fpr', param=0.3238036840257909)
        data_pre_processor = None  # VarianceThreshold()
        # data_pre_processor = GenericUnivariateSelect(mode='fdr', param=0.29998771459507223)

        # learner = ExtraTreesClassifier(criterion='entropy', max_features=0.32610721820585975, min_samples_leaf=13,
        #                                min_samples_split=5, n_estimators=512, warm_start=True)
        learner = ExtraTreesClassifier(bootstrap=True, criterion='entropy', max_features=0.1741962585712563,
                             min_samples_leaf=8, n_estimators=512, warm_start=True)
    else:
        print("Please provide the correct sensors.")
        exit(11)

    # load data
    X, y = load_tsfresh_feature(exp_names, sensors_names, split=True, dir=dir)
    X.dropna(axis=1, how='all', inplace=True)  # remove all nan value features (if any)
    if len(sensors) == 2:  # remove constant features for pn1 and pn3
        constant_columns = [col for col in X.columns if X[col].nunique() == 1]
        X.drop(columns=constant_columns, inplace=True)
        print(f"Removing constant features (in total {len(constant_columns)} feature(s)).")
        print(f"New shape: {X.shape}.")
    max_feature_set_size = X.shape[1]

    pl_interpretable = get_pipeline_for_features(learner, data_pre_processor, X, y, list(X.columns))

    df_auc_results_per_feature_combo = get_scores_for_feature_combinations(
        pl_interpretable,
        X,
        y,
        max_feature_set_size,
        dir,
        repeats_per_size={i: 100 for i in range(1, max_feature_set_size + 1)},
        # repeats_per_size={i: 10 for i in range(1, max_feature_set_size + 1)},
        num_combos_from_last_stage={i: 10 if i < 40 else (5 if i < 100 else 3) for i in range(2, max_feature_set_size + 1)},
        # num_combos_from_last_stage={i: 1 if i < 40 else (1 if i < 100 else 1) for i in
        #                             range(2, max_feature_set_size + 1)},
        sensors_names=sensors_names
    )

    k_s = list(range(1, len(df_auc_results_per_feature_combo) + 1))
    best_scores_per_k = []
    best_combos_per_k = []
    for k in k_s:
        df_fs = df_auc_results_per_feature_combo[k]
        best_scores_per_k.append(df_fs.iloc[0]["scores"])
        best_combos_per_k.append(df_fs.iloc[0]["combo"])
        print(k, np.mean(best_scores_per_k[-1]), np.std(best_scores_per_k[-1]))

    # plot best combos
    if plotting:
        fig, ax = plt.subplots(figsize=(10, 3))
        mu = np.array([np.mean(v) for v in best_scores_per_k])
        std = np.array([np.std(v) for v in best_scores_per_k])
        print(std)
        ax.plot(k_s, mu)
        ax.fill_between(k_s, mu - std, mu + std, alpha=0.2)
        for k, combo, score in zip(k_s, best_combos_per_k, mu):
            print("Chosen feature combinations for", k, score, str(combo))  # , rotation=90)
        ax.set_xlabel("Number of Features")
        ax.set_ylabel("AUC ROC")
        # ax.set_ylim([0.6, 0.8])
        ax.axhline(max(mu), color="black", linestyle="--")
        print(f"Best score {max(mu)} at {k_s[np.argmax(mu)]} features.")
        plt.show()

    ks_for_lcs = range(1, max_feature_set_size + 1)
    # ks_for_lcs = range(1, 300)

    lcs = {}  # learning classifier system for each k
    for k in ks_for_lcs:
        combo = best_combos_per_k[k - 1]
        lc_file = f"{dir}results/lcs/{sensors_names}_lcs_{k}.csv"
        # lc_file = f"{dir}results/lcs/lcs_{k}.csv"
        print(f"Get curves for {k} features with combo {combo}.")
        lcs[k] = get_learning_curves(
            learner=pl_interpretable,
            X=X[combo],
            y=y,
            repeats=500,
            first_anchor=0.05,
            last_anchor=0.9,
            steps=10,
            filename=lc_file,
            scoring="roc_auc"
        )
    # plot learning curves
    if plotting:
        fig, ax = plt.subplots(figsize=(16, 6))
        # ax.plot(schedule, lc[0].mean(axis=1), label="train AUC")
        for k in [1, 5, 10]:  # , 4, 8, 16]:
            schedule, lc = [float(v) for v in lcs[k].columns], lcs[k].values
            mu = lc.mean(axis=0)
            std = lc.std(axis=0)
            ax.plot(schedule, mu, label=f"{k} features")
            ax.fill_between(schedule, mu - std, mu + std, alpha=0.3)
        ax.set_title(f"Learning Curves for Validation ROC AUC")
        ax.legend()
        ax.set_xlim([0, 1.6])
        # ax.set_ylim([0.45,0.8])
        ax.axhline(0.725, color="blue", linestyle="--")
        ax.axhline(0.5, color="red", linestyle="--")
        plt.show()
