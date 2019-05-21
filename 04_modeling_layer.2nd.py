# Raw input + SVD input + randoem_seed + ...

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_squared_log_error

from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor


def get_log_loss(pred, actual):
  log_mse = mean_squared_error(np.log1p(pred), np.log1p(actual))
  result = np.sqrt(log_mse)
  return result

def get_loss(pred, actual):
  return np.sqrt(mean_squared_error(actual, pred))


# Import the all dataset
tr_1 = pd.read_csv('data/stack_train_1.csv')
te_1 = pd.read_csv('data/stack_test_1.csv')

tr_2 = pd.read_csv('data/stack_train_2.csv')
te_2 = pd.read_csv('data/stack_test_2.csv')

tr_3 = pd.read_csv('data/stack_train_3.csv')
te_3 = pd.read_csv('data/stack_test_3.csv')

X_tr = pd.concat([tr_1, tr_2, tr_3], axis = 0)     # y_val basis!
y_tr = X_tr.val
X_tr.drop('val', axis = 1, inplace = True)

te = pd.concat([te_1, te_2, te_3], axis = 0)

del tr_1, tr_2, tr_3, te_1, te_2, te_3

####################################################
# Second Layer
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
