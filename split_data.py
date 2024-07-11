from sklearn.model_selection import train_test_split
from utils.feature_loader import load_tsfresh_feature
import numpy as np

# todo specify which experiments and sensors to use
exp_names = ["Exp44_Ivy2", "Exp45_Ivy4", "Exp46_Ivy0", "Exp47_Ivy5"]  # "Exp44_Ivy2", "Exp45_Ivy4", "Exp46_Ivy0", "Exp47_Ivy5"
sensors = ["pn1", "pn3"]  # "pn1", "pn3"

x, y = load_tsfresh_feature(exp_names, sensors, clean=True)

x.reset_index(inplace=True)
y.reset_index(inplace=True)
rows_to_remove = np.where(x.isna().any(axis=1))[0]
y.drop(index=rows_to_remove, inplace=True)
x.dropna(inplace=True)
y.drop(columns="index", inplace=True)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# print(type(X_train), type(X_test), type(y_train), type(y_test))
# print(X_train.columns)

# expositions: 17, 19, 18, 24 = 78 samples --> 34, 38, 36, 48
# channel pn1 = 152 samples (30, 38, 36, 48)  -> 121 to 31 samples
# channel pn3 = 122 samples (30, 8, 36, 48)  -> 97 to 25 samples
# combined: 274 samples  -> 219 to 55 samples
# 274 samples in total

# X_train.to_csv(f"data_preprocessed/split_data/X_train_{sensors}.csv", index=False)
# X_test.to_csv(f"data_preprocessed/split_data/X_test_{sensors}.csv", index=False)
# y_train.to_csv(f"data_preprocessed/split_data/y_train_{sensors}.csv", index=False)
# y_test.to_csv(f"data_preprocessed/split_data/y_test_{sensors}.csv", index=False)