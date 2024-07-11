# WatchPlant Classification

## Ozone pollution classification
 

#### 1. Data preprocessing
Features can be calculated using `calc_features.py`. 
For calculating the features ts_fresh is used. 

`split_data.py` is used to split the data into training and test data.

#### 2. Find the best model

``find_classifier.py`` uses naive autoML to find the best model.
You can specify which feature setting should be used by setting `sensors` to
`["pn1"]` or `["pn3"]` or `["pn1", "pn3"]`.
The output should be the best model (based on the training data).

#### 3. Feature selection 

``learning_curve.py`` calculates the best feature combination using the roc auc scoring. 
Again you can choose the setting by passing the sensors via the command line. 

ATTENTION: This script probably takes a while to run.

#### 4. Learning curve

In a second step ``learning_curve.py`` creates the learning curve for the best feature combinations 
(for each number of features).
