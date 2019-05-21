import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.decomposition import TruncatedSVD
import operator

# Dimension Reduction
def dims_reductor(df, colname, a):
    svd = TruncatedSVD(n_components= a, random_state = 42)
    temp = svd.fit_transform(df)
    df = pd.DataFrame(temp)
    df.columns = [colname + '_' + str(i) for i in range(len(df.columns))]
    return df


# Load the data
tr = pd.read_csv('data/train_2.csv')
te = pd.read_csv('data/test_2.csv')

y = tr.revenue_log
tr.drop('revenue_log', axis = 1, inplace = True)

# combine the two data
cut = len(tr)
df = pd.concat([tr, te])


#######################################
# Label Encoding
## 1. status, is_collection, is_homepage
labelencoder = LabelEncoder()
df['status'] = labelencoder.fit_transform(df.status.astype('str'))
df['is_collection'] = labelencoder.fit_transform(df.is_collection)
df['is_homepage'] = labelencoder.fit_transform(df.is_homepage)

## 2. genres
df['genres'] = df.genres.map(lambda row: row.split(';')[:-1])
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df.genres).astype('int')
df_genres = pd.DataFrame(X, columns = '_genres_' + mlb.classes_)

## 3. production countries
df['production_countries'] = df.production_countries.map(lambda row: row.split(';')[:-1])
X = mlb.fit_transform(df.production_countries).astype('int')
df_prod_count = pd.DataFrame(X, columns ='_prod_count_' + mlb.classes_)

## 4. production companies
df['n_prod_comp'] = df.production_companies.apply(lambda row: row.count(';'))

df['production_companies'] = df.production_companies.map(lambda row: row.split(';')[:-1])
X = mlb.fit_transform(df.production_companies).astype('int')
df_prod_comp = pd.DataFrame(X, columns ='_prod_comp_' + mlb.classes_)

sub = df_prod_comp[:cut]
sub['revenue_log'] = y

# take only top 100 companies with significant gap in revenue
dict_comp = {}
for col in sub.columns:
    a = sub.revenue_log[sub[col] == 0].median()
    b = sub.revenue_log[sub[col] == 1].median()
    gap = np.abs(a-b)
    dict_comp[col] = gap

sorted_dict_comp = sorted(dict_comp.items(), key=operator.itemgetter(1), reverse=True)

gap_comp = []
for c_v in sorted_dict_comp[:100]:
    c, _ = c_v
    gap_comp.append(c)

df_prod_comp = df_prod_comp.loc[:, gap_comp]


## 5. spoken languages
df['spoken_languages'] = df.spoken_languages.apply(lambda row: row.split(';')[:-1])
X = mlb.fit_transform(df.spoken_languages).astype('int')
df_langs = pd.DataFrame(X, columns = '_spoken_lang_' + mlb.classes_)


## 6. crew_department
df.crew_department[df.crew_department.isnull()] = ''
df['crew_department'] = df.crew_department.apply(lambda row: row.split(';')[:-1])
X = mlb.fit_transform(df.crew_department).astype('int')
df_crew_dep = pd.DataFrame(X, columns = '_crew_' + mlb.classes_)



## Combine the encoded data
df.reset_index(drop = True, inplace = True)
df_all = pd.concat([df, df_genres, df_prod_count, df_prod_comp, df_langs, df_crew_dep], axis = 1)


# Drop features
drop_feats = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'original_language',
              'id', 'imdb_id', 'runtime', 'revenue', 'revenue.1', 'popularity', 'popularity2', 'totalVotes',
              'release_date', 'year_diff', 'quarter', 'budget',
              'n_cast', 'cast_gender', 'cast_female', 'cast_neutral',
              'n_crew', 'n_crew_department', 'crew_department',  'crew_job', 'n_crew_job', 'crew_gender', 'crew_female', 'crew_neutral']

df_all.drop(drop_feats, axis = 1, inplace = True)

print("============Final Column list (without multi-labels)============")
print(df_all.columns[:70])

tr = df_all[:cut]
te = df_all[cut:]
tr['revenue_log'] = y

tr.to_csv('data/train_3.csv', index = False)
te.to_csv('data/test_3.csv', index = False)


# Dimension Reduction
df_genres = dims_reductor(df_genres, 'genres',5)
df_prod_count = dims_reductor(df_prod_count, 'prod_count', 10)
df_prod_comp = dims_reductor(df_prod_comp, 'prod_comp', 10)
df_langs = dims_reductor(df_langs, 'langs', 10)
df_crew_dep = dims_reductor(df_crew_dep, 'crew_dep', 5)

df_all = pd.concat([df, df_genres, df_prod_count, df_prod_comp, df_langs, df_crew_dep], axis = 1)

tr = df[:cut]
te = df[cut:]
tr['revenue_log'] = y

tr.to_csv('data/train_3_svd.csv', index = False)
te.to_csv('data/test_3_svd.csv', index = False)
