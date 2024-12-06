import platform
import matplotlib
import pandas as pd
import naiveautoml
import logging
from sklearn.model_selection import train_test_split
from utils.feature_loader import load_tsfresh_feature
import matplotlib.pyplot as plt

# for interactive plots
if platform.system() == "Darwin":
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    plt.rcParams.update({'font.size': 20})
    pd.set_option('display.max_rows', None)
elif platform.system() == "Linux":
    matplotlib.use('TkAgg')
elif platform.system() == "Windows":
    # TODO
    pass


def plot_history(naml):
    scoring = naml.task.scoring["name"]

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(naml.history["time"], naml.history[scoring])
    ax.axhline(naml.history[scoring].max(), linestyle="--", color="black", linewidth=1)
    max_val = naml.history[scoring].max()
    median_val = naml.history[scoring].median()
    ax.set_ylim([median_val, max_val + (max_val - median_val)])
    plt.show()


if __name__ == "__main__":
    # do logging
    logger = logging.getLogger('naiveautoml')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # load data
    exp_names = ["Exp44_Ivy2", "Exp45_Ivy4", "Exp46_Ivy0", "Exp47_Ivy5"]
    sensors = ["pn1", "pn3"]  # "pn1", "pn3", "mu_ch1", "mu_ch2", "bnb_no_wind" (for bnb paper)
    sensors_names = "_".join(sensors)
    scoring = "roc_auc"  # "roc_auc", "accuracy"
    X_train, y_train = load_tsfresh_feature(exp_names, sensors_names, split=True)  # return the pre-split training data
    X_train.dropna(axis=1, how='all', inplace=True)  # remove all nan value features (if any)
    if len(sensors) == 2:  # remove constant features for pn1 and pn3
        constant_columns = [col for col in X_train.columns if X_train[col].nunique() == 1]
        X_train.drop(columns=constant_columns, inplace=True)
        print(f"Removing constant features (in total {len(constant_columns)} feature(s)).")
        print(f"New shape: {X_train.shape}.")

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy().ravel()

    naml = naiveautoml.NaiveAutoML(max_hpo_iterations=100, show_progress=True, scoring=scoring)
    naml.fit(X_train, y_train)

    print("---------------------------------")
    # print(naml.chosen_model)
    print("---------------------------------")
    print(naml.history)

    naml.history.to_csv(f"results/naml_history_{scoring}_{sensors_names}_new2.csv")
    plot_history(naml)