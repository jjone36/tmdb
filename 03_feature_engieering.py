import pandas as pd
import numpy as np

from sklearn.preprocessing import scale

# Import the all dataset
tr_num = pd.read_csv('data/train_num.csv')
tr_cat = pd.read_csv('data/train_cat.csv')

te_num = pd.read_csv('data/test_num.csv')
te_cat = pd.read_csv('data/test_cat.csv')

# Numberic features
drop_feats = ['id', 'budget', 'popularity', 'revenue']
tr_num.drop(drop_feats, axis = 1, inplace = True)
te_num.drop(drop_feats, axis = 1, inplace = True)

# Scaling
tr_num['runtime_m'] = scale(tr_num.runtime_m)
te_num['runtime_m'] = scale(te_num.runtime_m)


# Categorical features
cut = len(tr_cat)
df_cat = pd.concat([tr_cat, te_cat], axis = 0)

drop_feats = ['genres', 'imdb_id', 'production_companies', 'revenue', 'crew_job']
df_cat.drop(drop_feats, axis = 1, inplace = True)

# Label Encoding









# Combine the train set
tr_cat = df_cat[:cut]
te_cat = df_cat[cut:]

tr = pd.concat([tr_num, tr_cat], axis = 1)
te = pd.concat([te_num, te_cat], axis = 1)

# Save the files
tr.to_csv('data/train_2.csv', index = False)
te.to_csv('data/test_2.csv', index = False)
