import os
import platform
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import naiveautoml

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from utils.feature_loader import load_botanical
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from utils.learner_pipeline import get_pipeline_from_config
from sklearn.metrics import classification_report

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging


def fit_classifier_parallel(x_analysis, y_analysis, pl_interpretable, i_train, i_validation, n_classes=2):
    x_train = x_analysis.iloc[i_train]
    y_train = y_analysis.iloc[i_train]
    x_validation = x_analysis.iloc[i_validation]
    y_validation = y_analysis.iloc[i_validation]

    trained = clone(pl_interpretable).fit(x_train.values, y_train.values.ravel())
    y_pred = trained.predict(x_validation.values)
    y_pred_proba = trained.predict_proba(x_validation.values)

    # NOTE: ovo and macro insensitive to class inbalance for roc_auc, current solution is sensitive
    roc = roc_auc_score(y_validation, y_pred_proba[:, 1]) if n_classes == 2 else \
          roc_auc_score(y_validation, y_pred_proba, multi_class='ovr', average='macro')

    return accuracy_score(y_validation, y_pred), roc, \
           f1_score(y_validation, y_pred, average='weighted'), trained, None  # shap_value if use_shap else None


if __name__ == '__main__':
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


    stimuli_settings = [["cold", "warm"],
                        ["day", "night"],
                        ["dry", "rain"],
                        ["wind", "windless"]]

    channels = [["CH1"], ["CH2"]]
    number_of_repeats = 10
    n_classes = 2
    for stimuli in stimuli_settings:
        for channel in channels:
            config = f"results/2024_botanical_garden/autoML_classifiers/naml_history_{'_'.join(channel)}_tw_60_{'_'.join(stimuli)}_with_smote.csv"
            scoring = "f1_score"  # "accuracy"

            pl_interpretable = get_pipeline_from_config(config, scoring)


            x, y = load_botanical(stimuli, channel, path=path_to_data)

            X_analysis, X_test, y_analysis, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

            label_encoder = LabelEncoder()
            y_analysis = label_encoder.fit_transform(y_analysis["class"])
            y_test = label_encoder.transform(y_test["class"])
            y_analysis = pd.DataFrame(data=y_analysis, columns=["class"])
            y_test = pd.DataFrame(data=y_test, columns=["class"])
            # y_analysis = y_analysis["class"].to_numpy().ravel()
            # y_test = y_test["class"].to_numpy().ravel()

            # TODO SMOTE?
            from imblearn.over_sampling import SMOTE

            smote = SMOTE(random_state=42)
            X_analysis, y_analysis = smote.fit_resample(X_analysis, y_analysis)

            acc_all = []
            roc_all = []
            f1_all = []
            classifier_all = []
            pbar = tqdm(total=number_of_repeats)
            cv = StratifiedShuffleSplit(n_splits=number_of_repeats)
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [executor.submit(fit_classifier_parallel, X_analysis, y_analysis, pl_interpretable, i_train,
                                           i_validation, n_classes) for i_train, i_validation in
                           cv.split(X_analysis, y_analysis)]


                def _cb(future):
                    pbar.update(1)


                for future in futures:
                    future.add_done_callback(_cb)

                # await results
                as_completed([f for f in futures])

            # close pbar
            pbar.close()

            for i, future in enumerate(futures):
                acc, roc, f1, classifier, shap_value = future.result()
                acc_all.append(acc)
                roc_all.append(roc)
                f1_all.append(f1)
                classifier_all.append(classifier)

            test_accs = []
            test_rocs = []
            test_cm_00, test_cm_01, test_cm_02 = [], [], []
            test_cm_10, test_cm_11, test_cm_12 = [], [], []
            test_cm_20, test_cm_21, test_cm_22 = [], [], []

            for c in classifier_all:
                test_roc = roc_auc_score(y_test, c.predict_proba(X_test.values)[:, 1]) if n_classes == 2 else \
                    roc_auc_score(y_test, c.predict_proba(X_test.values), multi_class='ovr', average='macro')
                test_rocs.append(test_roc)
                test_acc = accuracy_score(y_test, c.predict(X_test.values))
                test_accs.append(test_acc)

                rep = classification_report(y_test, c.predict(X_test.values), output_dict=True)
                rep_df = pd.DataFrame(rep)
                rep_df.to_csv(f"results/2024_botanical_garden/report/{'_'.join(channel)}_tw_60_{'_'.join(stimuli)}_with_smote.csv", index=True)

                # NOTE: confusion matrix depends on number of classes
                if True:  # n_classes == 2:
                    test_cm = confusion_matrix(y_test, c.predict(X_test.values), labels=[0, 1])
                    test_cm_00.append(test_cm[0, 0])
                    test_cm_01.append(test_cm[0, 1])
                    test_cm_10.append(test_cm[1, 0])
                    test_cm_11.append(test_cm[1, 1])
                else:
                    test_cm = confusion_matrix(y_test, c.predict(X_test.values), labels=[0, 1, 2])
                    test_cm_00.append(test_cm[0, 0])
                    test_cm_01.append(test_cm[0, 1])
                    test_cm_02.append(test_cm[0, 2])
                    test_cm_10.append(test_cm[1, 0])
                    test_cm_11.append(test_cm[1, 1])
                    test_cm_12.append(test_cm[1, 2])
                    test_cm_20.append(test_cm[2, 0])
                    test_cm_21.append(test_cm[2, 1])
                    test_cm_22.append(test_cm[2, 2])

            if True: # n_classes == 2:
                save_frame = pd.DataFrame(data={"accuracy": acc_all, "roc_auc": roc_all, "f1_score": f1_all,
                                                "test_accuracy": test_accs, "test_roc_auc": test_rocs,
                                                "test_cm_00": test_cm_00, "test_cm_01": test_cm_01,
                                                "test_cm_10": test_cm_10, "test_cm_11": test_cm_11})
            else:
                save_frame = pd.DataFrame(data={"accuracy": acc_all, "roc_auc": roc_all, "f1_score": f1_all,
                                                "test_accuracy": test_accs, "test_roc_auc": test_rocs,
                                                "test_cm_00": test_cm_00, "test_cm_01": test_cm_01, "test_cm_02": test_cm_02,
                                                "test_cm_10": test_cm_10, "test_cm_11": test_cm_11, "test_cm_12": test_cm_12,
                                                "test_cm_20": test_cm_20, "test_cm_21": test_cm_21, "test_cm_22": test_cm_22})

            print("----------------------------------")
            print(f"Setting {stimuli}, {channel}")
            print(pl_interpretable)

            print(f"accuracy: {save_frame['test_accuracy'].mean():.4f} ($pm$ {save_frame['test_accuracy'].std():.4f})")
            print(f"f1: {save_frame['f1_score'].mean():.2f}")

            print(f"confusion matrix:\n{save_frame[['test_cm_00', 'test_cm_01', 'test_cm_10', 'test_cm_11']].mean()}")
            print("----------------------------------")




