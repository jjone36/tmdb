import pandas as pd
import numpy as np

# Import the data
tr = pd.read_csv('data/train.csv')
te = pd.read_csv('data/test.csv')

# Data overview
print("Data types : \n" , tr.info())
print("\nUnique values :  \n", tr.nunique())
print("\nMissing values :  ", tr.isnull().sum())

# Data overview
print("Data types : \n" , te.info())
print("\nUnique values :  \n", te.nunique())
print("\nMissing values :  ", te.isnull().sum())

# Separate categorical & numerical features
cat_feats = tr.columns[tr.dtypes == 'object']
num_feats = tr.columns[tr.dtypes != 'object']
print("Numeric variables ", len(num_feats))
print(num_feats.get_values())
print("Categorical variables : ", len(cat_feats))
print(cat_feats.get_values())

# Separate categorical & numerical features
y = tr.revenue

te_num = te[num_feats[:-1]]
te_cat = te[cat_feats]

tr_num = tr[num_feats]
tr_cat = tr[cat_feats]
tr_cat = pd.concat([tr_cat, y], axis = 1)

tr_num.to_csv('data/train_num.csv', encoding = 'utf-8-sig', index = False)
te_num.to_csv('data/test_num.csv', encoding = 'utf-8-sig', index = False)

tr_cat.to_csv('data/train_cat.csv', encoding = 'utf-8-sig', index = False)
te_cat.to_csv('data/test_cat.csv', encoding = 'utf-8-sig', index = False)
