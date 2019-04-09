import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

# Import the all dataset
tr = pd.read_csv('data/train_2.csv')
te = pd.read_csv('data/test_2.csv')

# Additional preprocessing (if neccesssary)




# Split into train and Valid set
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size = .2, random_state = 36)

# Modeling - Catboost
cat_idx
model_cb = CatBoostRegressor(iterations = 2,
                          depth = 2,
                          learning_rate = 1,
                          loss_function = 'Logloss',
                          logging_level = 'Verbose')
model_cb.fit(X_tr, y_tr, cat_features = cat_idx)




# Evaluation
