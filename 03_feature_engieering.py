import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
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

# Label Encoding
# 1. genres
df_cat['genres'] = df_cat.genres.apply(lambda row: row.split(';')[:-1])
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(tr_cat.genres).astype('int')
df_genres = pd.DataFrame(X, columns = mlb.classes_)
df_cat = pd.concat([df_cat, df_genres], axis = 1)

# 2. spoken languages
tr_cat['spoken_languages'] = tr_cat.spoken_languages.apply(lambda row: row.split(';'))
X = mlb.fit_transform(tr_cat.spoken_languages).astype('int')
df_langs = pd.DataFrame(X, columns = mlb.classes_)
# takes only top 20 languages
n_langs = df_langs.iloc[:, :-1].sum(axis = 0).astype('int').sort_values(ascending = False)
top20_langs = n_langs.head(20).keys()
df_cat = pd.concat([df_cat, df_langs.loc[:, top20_langs]], axis = 1)

# 3. n_cast, n_crew, n_crew_job
df_cat['n_cast_log'] = np.log1p(df_cat.n_cast)
df_cat['n_crew_log'] = np.log1p(df_cat.n_crew)
df_cat['n_crew_job_log'] = np.log1p(df_cat.n_crew_job)


# Drop features
drop_feats = ['genres', 'imdb_id', 'production_companies', 'revenue', 'crew_job', 'spoken_languages',
              'n_cast', 'n_crew', 'n_crew_job']
df_cat.drop(drop_feats, axis = 1, inplace = True)

# Combine the train set
tr_cat = df_cat[:cut]
te_cat = df_cat[cut:]

tr = pd.concat([tr_num, tr_cat], axis = 1)
te = pd.concat([te_num, te_cat], axis = 1)

# Save the files
tr.to_csv('data/train_2.csv', index = False)
te.to_csv('data/test_2.csv', index = False)
