import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


def get_loss(pred, actual):
  log_mse = mean_squared_error(np.log1p(pred), np.log1p(actual))
  result = np.sqrt(log_mse)
  return result


# Import the all dataset
tr = pd.read_csv('train_3.csv')
te = pd.read_csv('test_3.csv')

y = tr.revenue_log
X = tr.drop('revenue_log', axis = 1)

# Split into train and Valid set
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size = .2, random_state = 42)

print("The size of the train set ", X_tr.shape)
print("The size of the validation set ", X_val.shape)
print("The size of the test set ", te.shape)

del tr, X, y

# Create a dataframe for prediction
df = pd.DataFrame()
df['val'] = y_val


####################################################
# First Layer
# Xgboost
model_xg = xgb.XGBRegressor(objective = 'reg:linear',
                            n_estimators = 3000,
                            max_depth = 9,
                            learning_rate = 0.007,
                            early_stopping_rounds = 500,
                            gamma = 1.0,
                            #alpha = .6,
                            subsample = 0.7,
                            colsample_bytree = 0.7,
                            colsample_bylevel = 0.8,
                            silent = True)

model_xg.fit(X_tr, y_tr)
df['xg'] = model_xg.predict(X_val)

# Catboost
model_cb = CatBoostRegressor(bagging_temperature = 0.5,
                             colsample_bylevel = 0.8,
                             depth = 9,
                             early_stopping_rounds = 500,
                             eval_metric = 'RMSE',
                             iterations = 3000,
                             learning_rate = .05,
                             logging_level = 'Silent')

model_cb.fit(X_tr, y_tr, eval_set = (X_val, y_val), use_best_model = True)
df['cb'] = model_cb.predict(X_val)

# LightGBM
tr_data = lgb.Dataset(X_tr, label = y_tr)
val_data = lgb.Dataset(X_val, label = y_val)

params = {'objective' : 'regression',
          'num_iterations' : 10000,
          'max_depth' : 13,
          'num_leaves' : 100,
          'learning_rate': 0.003,
          'metric' : 'rmse',
          'min_data_in_leaf' : 100,
          'colsample_bytree': 0.8,
          'subsample_freq': 1,
          'lambda_l1' : 0.01,
          'lambda_l2' : 0.5,
          'subsample' : 0.8,
          'verbose' : -1}

hist = {}
model_lg = lgb.train(params, tr_data,
                     valid_sets = [val_data],
                     verbose_eval = -1,
                     early_stopping_rounds = 500,
                     callbacks = [lgb.record_evaluation(hist)])

df['lg'] = model_lg.predict(X_val, num_iteration = model_lg.best_iteration)


# Keras
model = Sequential()

model.add(Dense(input_dim = X_tr.shape[1], output_dim = 128, activation = 'relu'))
model.add(Dense(output_dim = 64, activation = 'relu'))
model.add(Dense(output_dim = 16, activation = 'relu'))
model.add(Dense(output_dim = 1))

model.compile(optimizer = Adam(lr = .1),
              loss = 'mse',
              metrics = ['mean_squared_error'])

early_stopper = EarlyStopping(patience = 5)
r = model.fit(X_tr, y_tr,
              validation_data = (X_val, y_val),
              batch_size = 300,
              nb_epoch = 1000,
              callbacks = [early_stopper])

model_ke = model
df['ke'] = model.predict(X_val)

# Make the first prediction data
prediction = pd.DataFrame()
model_list = [model_xg, model_cb, model_lg, model_ke]
for model in model_list:
    name = str(model)[-2]
    pred = model.predict(te)
    prediction[name] = pred

####################################################
# Second Layer
X_tr_2 = prediction
from sklearn.linear_model import LinearRegression
model2_lr = LinearRegression()
model2_lr.fit(X_tr_2, y)

from sklearn.svm import LinearSVR
model2_svm = LinearSVR(epsilon=1.)
model2_svm.fit(X_tr_2, y)

from sklearn.ensemble import RandomForestRegressor
model2_rf = RandomForestRegressor(n_estimators = 100, max_depth = 3)
model2_rf.fit(X_tr_2, y)


####################################################
# Third Layer
