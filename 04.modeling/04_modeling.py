
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error

from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm.sklearn import LGBMRegressor

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping


# Loss funcion
def get_log_loss(pred, actual):
  log_mse = mean_squared_error(np.log1p(pred), np.log1p(actual))
  result = np.sqrt(log_mse)
  return result


# Kfold validation and creating meta-data
def kfold_validate(X, y, te, model, col):
    '''
    Implementing Kfold cross validation for each 1st layer model
    input: - X, y, te: the train and test set (input)
           - model: a model for training
           - col: column name in the meta-data
    output: - meta_tr, meta_te : the meta-data with the predicted values
    '''
    scores = np.zeros(k)
    for i, (tr_idx, val_idx) in enumerate(cv.split(X)):

        # Split the train and validation set
        X_tr, y_tr = X.iloc[tr_idx, :], y[tr_idx]
        X_val, y_val = X.iloc[val_idx, :], y[val_idx]

        # Train the model for each validation set
        model.fit(X_tr, y_tr)
        pred_val = model.predict(X_val)
        scores[i] = get_loss(pred_val, y_val)
        print("========={}-th Fold Score: {}".format(i, scores[i]))

        meta_tr[col][val_idx] = pred_val
        meta_tr['val'][val_idx] = y_val

    # Train the model for entire trainset and predict
    print("=========Total Score: ", np.mean(scores))
    model.fit(X, y)
    meta_te[col] = model.predict(te)



dir = '../'
# Import the all dataset
tr = pd.read_csv(dir + 'data/train_3.csv')
te = pd.read_csv(dir + 'data/test_3.csv')

# Final check
print("The size of the train set ", tr.shape)
print("The size of the test set ", te.shape)

print("NAs in train set\n", tr.isnull().sum()[tr.isnull().sum() != 0])
print("NAs in test set\n", te.isnull().sum()[te.isnull().sum() != 0])


y = tr.revenue_log
X = tr.drop('revenue_log', axis = 1)

# Create meta-trainset
cols = ['el', 'knn', 'rf', 'rf2', 'xg', 'cb', 'cb2', 'lg', 'ke']

nrow = X.shape[0]
ncol = len(cols)
dims = np.zeros((nrow, ncol))

meta_tr = pd.DataFrame(dims, columns = cols)
meta_tr['val'] = 0

# Create meta-testset
nrow = te.shape[0]
dims = np.zeros((nrow, ncol))
meta_te = pd.DataFrame(dims, columns = cols)

# Cross Validation
k = 10
seed = 10
cv = KFold(n_splits = k, shuffle = True, random_state = seed)


####################################################
######### 1st layer
####################################################
# 1. Elastic Net
model_el = ElasticNet(alpha = .005, l1_ratio = 0, max_iter = 5000)
kfold_validate(X, y, te, model_el, 'el')


# 2. KNN
model_knn = KNeighborsRegressor(n_neighbors = 10, weights = 'distance', p = 2)
kfold_validate(X, y, te, model_knn, 'knn')


# 3. Random Forest
model_rf = RandomForestRegressor(n_estimators = 3000,
                                 criterion = 'mse',
                                 max_depth = 9)
kfold_validate(X, y, te, model_rf, 'rf')

model_rf = RandomForestRegressor(n_estimators = 3000,
                                 criterion = 'mse',
                                 max_depth = 11)
kfold_validate(X, y, te, model_rf, 'rf_2')


# 4. Kernel SVM
model_svm = SVR(kernel = 'rbf')
kfold_validate(X, y, te, model_svm, 'svm')


# 5. Xgboost
model_xg = XGBRegressor(objective = 'reg:linear',
                            n_estimators = 3000,
                            max_depth = 11,
                            learning_rate = 0.01,
                            early_stopping_rounds = 500,
                            gamma = 1.0,
                            #alpha = .6,
                            subsample = 0.7,
                            colsample_bytree = 0.6,
                            colsample_bylevel = 0.5,
                            silent = True)
kfold_validate(X, y, te, model_xg, 'xg')


# 6. Catboost
model_cb = CatBoostRegressor(bagging_temperature = 0.3,
                             colsample_bylevel = 0.7,
                             depth = 9,
                             early_stopping_rounds = 500,
                             eval_metric = 'RMSE',
                             iterations = 3000,
                             learning_rate = .01,
                             logging_level = 'Silent')
kfold_validate(X, y, te, model_cb, 'cb')

model_cb = CatBoostRegressor(bagging_temperature = 0.3,
                             colsample_bylevel = 0.7,
                             depth = 9,
                             early_stopping_rounds = 500,
                             eval_metric = 'RMSE',
                             iterations = 3000,
                             learning_rate = .05,
                             logging_level = 'Silent')
kfold_validate(X, y, te, model_cb, 'cb2')


# 7. LightGBM
params = {'objective' : 'regression',
          'num_iterations' : 5000,
          'max_depth' : 9,
          'num_leaves' : 100,
          'learning_rate': 0.005,
          'metric' : 'rmse',
          'min_data_in_leaf' : 100,
          'colsample_bytree': 0.5,
          'subsample_freq': 1,
          'lambda_l1' : 0.01,
          'lambda_l2' : 0.7,
          'subsample' : 0.8,
          'verbose' : -1}

scores = np.zeros(10)
for i, (tr_idx, val_idx) in enumerate(cv.split(X)):
    X_tr, y_tr = X.iloc[tr_idx, :], y[tr_idx]
    X_val, y_val = X.iloc[val_idx, :], y[val_idx]

    tr_data = lgb.Dataset(X_tr, label = y_tr)
    val_data = lgb.Dataset(X_val, label = y_val)

    hist = {}
    model_lg = lgb.train(params, tr_data,
                         valid_sets = [val_data],
                         verbose_eval = -1,
                         early_stopping_rounds = 500,
                         callbacks = [lgb.record_evaluation(hist)])

    pred_val = model_lg.predict(X_val, num_iteration = model_lg.best_iteration)
    scores[i] = get_loss(pred_val, y_val)
    print("========={}-th Fold Score: {}".format(i, scores[i]))

    meta_tr['lg'][val_idx] = pred_val

print("=========Total Score: ", np.mean(scores))
meta_te['lg'] = model_lg.predict(te)


# 8. Keras
# initialize the model
model = Sequential()

model.add(Dense(input_dim = X.shape[1], output_dim = 128, activation = 'relu'))
model.add(Dense(output_dim = 64, activation = 'relu'))
#model.add(Dropout(.7))
model.add(BatchNormalization())
model.add(Dense(output_dim = 32, activation = 'relu'))
#model.add(Dropout(.7))
model.add(BatchNormalization())
model.add(Dense(output_dim = 16, activation = 'relu'))
model.add(Dense(output_dim = 1))

# compile: stochastic gradient descent
model.compile(optimizer = Adam(lr=0.01),
              loss = 'mse',
              metrics = ['mean_squared_error'])

# early stopper
early_stopper = EarlyStopping(patience = 5)


results = np.zeros((3000, 1))
scores = np.zeros(10)
for i, (tr_idx, val_idx) in enumerate(cv.split(X)):

    # Split the train and validation set
    X_tr, y_tr = X.iloc[tr_idx, :], y[tr_idx]
    X_val, y_val = X.iloc[val_idx, :], y[val_idx]

    # Train the model for each validation set
    model_ke.fit(X_tr, y_tr,
                validation_data = (X_val, y_val),
                batch_size = 300,
                nb_epoch = 100,
                callbacks = [early_stopper],
                verbose= -1)
    pred_val = model_ke.predict(X_val)
    scores[i] = get_loss(pred_val, y_val)
    print("========={}-th Fold Score: {}".format(i, scores[i]))
    results[val_idx] = pred_val

# Train the model for entire trainset and predict
meta_tr['ke'] = results
print("=========Total Score: ", np.mean(scores))

model_ke.fit(X, y, batch_size = 100, nb_epoch = 100)
meta_te['ke'] = model_ke.predict(te)


#meta_tr.to_csv('meta_tr.csv', index = False)
#meta_te.to_csv('meta_te.csv', index = False)


####################################################
######### 2nd layer
####################################################
print("The size of the train set ", meta_tr.shape)
print("The size of the test set ", meta_te.shape)

y = meta_tr.org_y
X = meta_tr.drop('org_y', axis = 1)
te = meta_te

# Create meta-train set
cols = ['lr', 'el', 'rd', 'svm', 'knn', 'rf']

nrow = X.shape[0]
ncol = len(cols)
dims = np.zeros((nrow, ncol))

meta_tr = pd.DataFrame(dims, columns = cols)
meta_tr['val'] = 0

# Create meta-test set
nrow = te.shape[0]
dims = np.zeros((nrow, ncol))
meta_te = pd.DataFrame(dims, columns = cols)

# Cross Validation
k = 5
seed = 10
cv = KFold(n_splits = k, shuffle = True, random_state = seed)

# 1. Lienar Regression
model_lr = LinearRegression(normalize=True)
kfold_validate(X, y, te, model_lr, 'lr')     # 1.8395


# 2. ElasticNet
model_el = ElasticNet(alpha = .005, l1_ratio = 0, max_iter = 5000)
kfold_validate(X_tr, y, te, model_el, 'el')   # 1.8383


# 3. Ridge Regression
model_rd = Ridge(alpha = .001, normalize = True)
kfold_validate(X_tr, y, te, model_rd, 'rd')   # 1.8383


# 4. KNN
model_knn = KNeighborsRegressor(n_neighbors = 50, weights = 'distance', p = 2)
kfold_validate(X_tr, y, te, model_knn, 'knn')    # 1.8870


# 5. Random Forest (max_depth = 3)
model_rf = RandomForestRegressor(n_estimators = 1000, max_depth = 3)
kfold_validate(X_tr, y, te, model_rf, 'rf')   # 1.8502


# 6. Random Forest (max_depth = 5)
model_rf = RandomForestRegressor(n_estimators = 1000, max_depth = 5)
kfold_validate(X_tr, y, te, model_rf, 'rf2')    # 1.8489


# 7. Adatboost Regressor
model_ad = AdaBoostRegressor(n_estimators = 1000, learning_rate = .05,
                             loss = 'square', random_state = seed)
kfold_validate(X_tr, y, te, model_rf, 'ad')    # 1.8474



####################################################
######### 3rd layer
####################################################
y = meta_tr.val
X_tr = meta_tr.drop('val', axis = 1)
te = meta_te

# Create meta-train set
cols = ['lr', 'rd', 'rf', 'rf2']

nrow = X.shape[0]
ncol = len(cols)
dims = np.zeros((nrow, ncol))

meta_tr = pd.DataFrame(dims, columns = cols)
meta_tr['val'] = 0

# Create meta-test set
nrow = te.shape[0]
dims = np.zeros((nrow, ncol))
meta_te = pd.DataFrame(dims, columns = cols)

# Cross Validation
k = 3
seed = 10
cv = KFold(n_splits = k, shuffle = True, random_state = seed)

# 1. Linear Regression
model_lr = LinearRegression()
kfold_validate(X_tr, y, te, model_lr, 'lr')    # 1.8417

# 2. Ridge Regression
model_rd = Ridge(alpha = .01, normalize = True)
kfold_validate(X_tr, y, te, model_rd, 'rd')     # 1.8353

# 3. Random Forest
model_rf = RandomForestRegressor(n_estimators = 3000, max_depth = 3)
kfold_validate(X_tr, y, te, model_rf, 'rf')     # 1.8246

# 4. Random Forest
model_rf = RandomForestRegressor(n_estimators = 2000, max_depth = 5)
kfold_validate(X_tr, y, te, model_rf, 'rf2')    # 1.8149


fig, axes = plt.subplots(1, 4, figsize = (12, 5))
for i, col in enumerate(meta_tr.columns[meta_tr.columns != 'val']):
  sns.regplot(x = col, y = 'val', data = meta_tr, ax = axes[i%4])


final = meta_te.rd    # Final Submission

# Submission
sub = pd.read_csv(dir + 'data/sample_submission.csv')
sub['revenue'] = np.expm1(final)

name = 'final_sub'
sub.to_csv(dir + 'data/sub/' + name + '.csv', index = False)
