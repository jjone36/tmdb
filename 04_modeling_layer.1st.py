# Raw input + SVD input + randoem_seed + ...

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_squared_log_error

#from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping



def get_log_loss(pred, actual):
  log_mse = mean_squared_error(np.log1p(pred), np.log1p(actual))
  result = np.sqrt(log_mse)
  return result

def get_loss(pred, actual):
  return np.sqrt(mean_squared_error(actual, pred))

seed = 42

# Import the all dataset
tr = pd.read_csv('train_3.csv')
te = pd.read_csv('test_3.csv')

# Final check for NAs
print("NAs in train set\n", tr.isnull().sum()[tr.isnull().sum() != 0])
print("NAs in test set\n", te.isnull().sum()[te.isnull().sum() != 0])


y = tr.revenue_log
X = tr.drop('revenue_log', axis = 1)

# Dimension Reduction
svd = TruncatedSVD(n_components=5, random_state = seed)
X_svd = svd.fit(X)
X_svd = pd.DataFrame(X_svd.fit_transform(X))

svd_ratio = pd.DataFrame(X_svd.explained_variance_ratio_)
sns.barplot(x = svd_ratio.index, y = svd_ratio[0])


# Split into train and Valid set
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size = .3, random_state = seed)

print("The size of the train set ", X_tr.shape)
print("The size of the validation set ", X_val.shape)
print("The size of the test set ", te.shape)

del tr, X, y

# Create a dataframe for prediction
pred_l1 = pd.DataFrame()
pred_l1['val'] = y_val

####################################################
# First Layer
# 1. Elastic Net
model_el = ElasticNet(alpha = .3, l1_ratio = 0, max_iter = 5000)
model_el.fit(X_tr, y_tr
pred_l1['el'] = model_el.predict(X_val)


# 2. KNN
model_knn = KNeighborsRegressor(n_neighbors = 200,
                                weights = 'distance',
                                p = 2)
model_knn.fit(X_tr, y_tr
pred_l1['knn'] = model_knn.predict(X_val)


# 3. Random Forest
model_rf = RandomForestRegressor(n_estimators = 3000,
                                 criterion = 'mse',
                                 #min_samples_split = .7,
                                 #min_samples_leaf = 100,
                                 max_depth = 9)
model_rf.fit(X_tr, y_tr)
pred_l1['rf'] = model_rf.predict(X_val)


# 4. Kernel SVM
model_svm = SVR(kernel = 'rbf')
model_svm.fit(X_tr, y_tr)
pred_l1['svm'] = model_svm.predict(X_val)


# 5. Xgboost
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
pred_l1['xg'] = model_xg.predict(X_val)


## Save the model
#filename = 'xg_model.pkl'
#with open(filename, 'wb') as file:
#  pickle.dump(model_xg, file)


# 6. Catboost
model_cb = CatBoostRegressor(bagging_temperature = 0.5,
                             colsample_bylevel = 0.8,
                             depth = 9,
                             early_stopping_rounds = 500,
                             eval_metric = 'RMSE',
                             iterations = 3000,
                             learning_rate = .05,
                             logging_level = 'Silent')

model_cb.fit(X_tr, y_tr,
            eval_set = (X_val, y_val),
            use_best_model = True)
pred_l1['cb'] = model_cb.predict(X_val)

## Save the model
#filename = 'cb_model.pkl'
#with open(filename, 'wb') as file:
#  pickle.dump(model_cb, file)

# 7. LightGBM
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

pred_l1['lg'] = model_lg.predict(X_val, num_iteration = model_lg.best_iteration)


# 8. Keras
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
pred_l1['ke'] = model.predict(X_val)


# Make the first prediction data
te_l1 = pd.DataFrame()
model_list = [model_el, model_knn, model_rf, model_svm,
             model_xg, model_cb, model_lg, model_ke]
for model in model_list:
    name = str(model)[-2]
    pred = model.predict(te)
    te_l1[name] = pred



####################################################
# Second Layer
y_tr = pred_l1.val
X_tr = pred_l1.drop('val', axis = 1)
te = te_l1

# 1. Lienar Regression
model2_lr = LinearRegression()
model2_lr.fit(X_tr, y_tr)
pred_lr = model2_lr.predict(te)

# 2. Lienar SVM
model2_svm = LinearSVR(epsilon=1.)
model2_svm.fit(X_tr, y_tr)
pred_svm = model2_svm.predict(te)

# 3. Shallow Random Forest
model2_rf = RandomForestRegressor(n_estimators = 100, max_depth = 3)
model2_rf.fit(X_tr, y_tr)
pred_rf = model2_rf.predict(te)


####################################################
# Third Layer
sub = pd.read_csv('sample_submission.csv')

pred_final = pred_lr*.3 + pred_svm*.3 + pred_rf*.
sub['revenue'] = np.expm1(pred_final)

sub.to_csv('submission_!.csv', index = False)
