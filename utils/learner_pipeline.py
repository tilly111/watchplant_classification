from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, get_scorer
import shap
import pandas as pd
import numpy as np
import re


# def get_pipeline_for_features(classifier, X, y, feature_list):
def get_pipeline_for_features(classifier, data_pre_processor, X=None, y=None, feature_list=None):
    steps = []
    # TODO we need no encoding
    # attributes_that_require_encoding = list(set(feature_list) & set(categorical_attributes))
    # if attributes_that_require_encoding:
    #     steps.append(("Categorical Encoder", make_column_transformer((OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), attributes_that_require_encoding), remainder="passthrough")))
    if data_pre_processor is not None:
        steps.append(('data-pre-processor', data_pre_processor))
    steps.append(("learner", classifier))
    return Pipeline(steps)

def get_pipeline_from_config(config: str, scoring: str):
    config = pd.read_csv(config)
    # sort config by accuracy
    config = config.sort_values(by=scoring, ascending=False)
    pipeline_config = config["pipeline"].iloc[0]
    data_preprocessor_class = config["data-pre-processor_class"].iloc[0] if config["data-pre-processor_class"].iloc[0] is not np.nan else "sklearn.preprocessing._data.MinMaxScaler"
    feature_preprocessor_class = config["feature-pre-processor_class"].iloc[0] if config["feature-pre-processor_class"].iloc[0] is not np.nan else "sklearn.decomposition._pca.PCA"
    learner_class = config["learner_class"].iloc[0]

    # in case of select percentile we need to give a function which seems to be buggy
    pattern = r"<function chi2[^>]*>"
    pipeline_config = re.sub(pattern, "f_classif", pipeline_config)
    pipeline_config = "from sklearn.pipeline import Pipeline \n" \
                      f"from {'.'.join([part for part in data_preprocessor_class.split('.') if not part.startswith('_')][:-1])} import {data_preprocessor_class.split('.')[-1]} \n" \
                      f"from {'.'.join([part for part in feature_preprocessor_class.split('.') if not part.startswith('_')][:-1])} import {feature_preprocessor_class.split('.')[-1]} \n" \
                      f"from {'.'.join([part for part in learner_class.split('.') if not part.startswith('_')][:-1])} import {learner_class.split('.')[-1]} \n" \
                      "pipe=" + pipeline_config
    scope = {}
    exec(pipeline_config, scope)
    pipe = scope["pipe"]
    return pipe


def fit_classifier(learner, x_train, x_test, y_train, y_test, scoring="accuracy", use_shap=False, n_classes=2):
    learner_c = clone(learner).fit(x_train.to_numpy(), y_train.values.ravel())
    y_pred = learner_c.predict(x_test.values)

    shap_values = pd.DataFrame(data=np.zeros((1, x_train.shape[1])), columns=x_train.columns)
    if use_shap:
        # explainer = shap.KernelExplainer(learner_c.predict_proba, shap.sample(x_train, 100))  # x_test or x_train?
        explainer = shap.TreeExplainer(learner_c)
        shap_values = explainer.shap_values(x_test.to_numpy())  #, check_additivity=False
        # shap_value.values = shap_value.values[:, :, 1]
        # shap_value.base_values = shap_value.base_values[:, 1]
        # shap_values[:] = shap_value.abs.mean(axis=0).values
        # instance_idx = 0
        # shap.force_plot(explainer.expected_value[1], shap_values[1][instance_idx, :], x_test.iloc[instance_idx, :],
        #                 feature_names=x_test.columns)
        # plt.show()

    scorer = get_scorer(scoring)  # roc_auc
    c_m = confusion_matrix(y_test, y_pred, labels=range(n_classes))
    # return accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred), pl_interpretable
    return scorer(learner_c, x_test.values, y_test), c_m, shap_values  # , pl_interpretable


def fit_classifier_cf(learner, x_train, x_test, y_train, y_test, scoring="accuracy", use_shap=False):
    learner_c = clone(learner).fit(x_train.to_numpy(), y_train.values.ravel())
    y_pred = learner_c.predict(x_test.values)

    shap_values = pd.DataFrame(data=np.zeros((1, x_train.shape[1])), columns=x_train.columns)
    if use_shap:
        explainer = shap.KernelExplainer(learner_c.predict_proba, shap.sample(x_train, 100))  # x_test or x_train?
        shap_value = explainer(x_test)
        shap_value.values = shap_value.values[:, :, 1]
        shap_value.base_values = shap_value.base_values[:, 1]
        shap_values[:] = shap_value.abs.mean(axis=0).values

    scorer = get_scorer(scoring)  # roc_auc

    # return accuracy_score(y_test, y_pred), confusion_matrix(y_test, y_pred), pl_interpretable
    return scorer(learner_c, x_test.values, y_test), confusion_matrix(y_test, y_pred), shap_values, learner_c

