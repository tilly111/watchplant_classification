from sklearn.pipeline import Pipeline


def get_pipeline_for_features(classifier, X, y, feature_list):
    steps = []
    # TODO we need no encoding
    # attributes_that_require_encoding = list(set(feature_list) & set(categorical_attributes))
    # if attributes_that_require_encoding:
    #     steps.append(("Categorical Encoder", make_column_transformer((OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), attributes_that_require_encoding), remainder="passthrough")))
    steps.append(("Learner", classifier))
    return Pipeline(steps)