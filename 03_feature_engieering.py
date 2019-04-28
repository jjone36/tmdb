import pandas as pd
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, scale
import operator

# Import the all dataset
tr_num = pd.read_csv('data/train_num_p.csv')
tr_cat = pd.read_csv('data/train_cat_p.csv')

te_num = pd.read_csv('data/test_num_p.csv')
te_cat = pd.read_csv('data/test_cat_p.csv')

############################################
# Numberic features
# Scaling
tr_num['runtime_m'] = scale(tr_num.runtime_m)
te_num['runtime_m'] = scale(te_num.runtime_m)

# Drop the features
drop_feats = ['id', 'budget', 'popularity', 'runtime']
te_num.drop(drop_feats, axis = 1, inplace = True)
tr_num.drop(drop_feats, axis = 1, inplace = True)
tr_num.drop('revenue', axis = 1, inplace = True)



############################################
# Categorical features
cut = len(tr_cat)
df_cat = pd.concat([tr_cat, te_cat], axis = 0)

# Label Encoding
## 1. status, is_collection, is_homepage
labelencoder = LabelEncoder()
df_cat['status'] = labelencoder.fit_transform(df_cat.status.astype('str'))
df_cat['is_collection'] = labelencoder.fit_transform(df_cat.is_collection)
df_cat['is_homepage'] = labelencoder.fit_transform(df_cat.is_homepage)

## 2. genres
df_cat['genres'] = df_cat.genres.map(lambda row: row.split(';')[:-1])
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df_cat.genres).astype('int')
df_genres = pd.DataFrame(X, columns = 'genres_' + mlb.classes_)

## 3. production countries
df_cat['production_countries'] = df_cat.production_countries.map(lambda row: row.split(';')[:-1])
X = mlb.fit_transform(df_cat.production_countries).astype('int')
df_prod_count = pd.DataFrame(X, columns ='prod_count_' + mlb.classes_)

## 4. production companies
df_cat['n_prod_comp'] = df_cat.production_companies.apply(lambda row: row.count(';'))

df_cat['production_companies'] = df_cat.production_companies.map(lambda row: row.split(';')[:-1])
X = mlb.fit_transform(df_cat.production_companies).astype('int')
df_prod_comp = pd.DataFrame(X, columns ='prod_count_' + mlb.classes_)

# take only top 100 companies with significant gap in revenue
dict_comp = {}
for col in df_prod_comp.columns:
    a = df_prod_comp.revenue_log[df_prod_comp[col] == 0].median()
    b = df_prod_comp.revenue_log[df_prod_comp[col] == 1].median()
    gap = np.abs(a-b)
    dict_comp[col] = gap

sorted_dict_comp = sorted(dict_comp.items(), key=operator.itemgetter(1), reverse=True)

gap_comp = []
for c_v in sorted_dict_comp[:100]:
    c, _ = c_v
    gap_comp.append(c)

df_prod_comp = df_prod_comp.loc[:, gap_comp]


## 5. spoken languages
df_cat['spoken_languages'] = df_cat.spoken_languages.apply(lambda row: row.split(';')[:-1])
X = mlb.fit_transform(df_cat.spoken_languages).astype('int')
df_langs = pd.DataFrame(X, columns = 'spoken_lang_' + mlb.classes_)


## 6. crew_department
df_cat.crew_department[df_cat.crew_department.isnull()] = ''
df_cat['crew_department'] = df_cat.crew_department.apply(lambda row: row.split(';')[:-1])
X = mlb.fit_transform(df_cat.crew_department).astype('int')
df_crew_dep = pd.DataFrame(X, columns = 'crew_' + mlb.classes_)

## 7. crew_job (???)


## Combine the encoded data
#df_cat.reset_index(drop = True, inplace = True)
df_2 = pd.concat([df_genres, df_prod_count, df_prod_comp, df_langs, df_crew_dep], axis = 1)

# n_cast, n_crew, n_crew_job
df_cat['n_cast_log'] = np.log1p(df_cat.n_cast)
df_cat['n_crew_log'] = np.log1p(df_cat.n_crew)
#df_cat['n_crew_job_log'] = np.log1p(df_cat.n_crew_job)    -> remove n_crew_job

# cast, crew genders
df_cat['cast_male'] /= df_cat.n_cast
df_cat['crew_male'] /= df_cat.n_crew

df_cat.n_crew_profile[df_cat.n_crew_profile.isnull()] = 0
df_cat['n_crew_profile'] /= df_cat.n_crew

# Drop features
drop_feats = ['genres', 'production_companies', 'production_countries', 'revenue', 'spoken_languages', 'original_languages', 'release_date',
              'n_cast', 'n_crew', 'crew_department',  'n_crew_department', 'crew_job', 'n_crew_job',
              'cast_gender', 'cast_female', 'cast_neutral', 'crew_gender', 'crew_female', 'crew_neutral']
df_cat.drop(drop_feats, axis = 1, inplace = True)

# Combine the train set
tr_cat = df_cat[:cut]
te_cat = df_cat[cut:]

tr_cat_2 = df_2[:cut]
te_cat_2 = df_2[cut:]
te_cat_2.reset_index(drop = True, inplace = True)

tr = pd.concat([tr_num, tr_cat, tr_cat_2], axis = 1)
te = pd.concat([te_num, te_cat, te_cat_2], axis = 1)

# Save the files
tr.to_csv('data/train_2.csv', index = False)
te.to_csv('data/test_2.csv', index = False)
