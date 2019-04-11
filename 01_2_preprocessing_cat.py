import pandas as pd
import numpy as np

import ast
from collections import Counter
from datetime import date

# IMDB API
# https://imdbpy.sourceforge.io/
from imdb import IMDb
ia = IMDb()

# dictionary form features
def list_to_dict(row, a):
    # Convert to dict
    dict_list = ast.literal_eval(row)
    temp = ''
    for d in dict_list:
        if(a in d.keys()):
            temp += d[a] + ';'
    return temp

def dict_to_col(col, key_name):
    # Fill NAs
    df_cat[col] = df_cat[col].fillna('[{}]')
    # Get the element list
    df_cat[col] = df_cat[col].apply(lambda row: list_to_dict(row, key_name))


# Import the categorical sub data
tr = pd.read_csv('data/train_cat.csv')
te = pd.read_csv('data/test_cat.csv')

# Concat train and test set
cut = len(tr)
df_cat = pd.concat([tr, te], axis = 0).reset_index(drop = True)


##### belongs to collection
df_cat['is_collection'] = df_cat.belongs_to_collection.notnull()

##### homepage
df_cat['is_homepage'] = df_cat.homepage.notnull()


##### genres
dict_to_col('genres', 'name')

# fill NAs in genres
df_cat.genres[470] = 'Adventure;'
df_cat.genres[1622] = 'Drama;Comedy;'
df_cat.genres[1814] = 'Comedy;'
df_cat.genres[1819] = 'Romance;Drama;'
df_cat.genres[2423] = 'Action;Drama;Romance;'
df_cat.genres[2686] = 'Thriller;'
df_cat.genres[2900] = 'Drama;Fantasy;Mystery;'
df_cat.genres[3073] = 'Drama;Fantasy;Mystery;'
df_cat.genres[3793] = 'Drama;Romance;'
df_cat.genres[3910] = 'Adventure;Biography;Drama;'
df_cat.genres[4221] = 'Action;Crime;Drama;'
df_cat.genres[4442] = 'Drama;'
df_cat.genres[4615] = 'Comedy;'
df_cat.genres[4964] = 'Action;Crime;'
df_cat.genres[5062] = 'Action;Drama;'
df_cat.genres[5118] = 'Drama;'
df_cat.genres[5213] = 'Documentary;Biography;Family;'
df_cat.genres[5251] = 'Comedy;'
df_cat.genres[5519] = 'Comedy;Crime;Mystery;'
df_cat.genres[6449] = 'Comedy;Crime;Drama;'
df_cat.genres[6485] = 'Musical;Comedy;'
df_cat.genres[6564] = 'Documentary;Drama;War;'
df_cat.genres[6817] = 'Drama;'

# number of genres
df_cat['n_genres'] = df_cat.genres.apply(lambda row: row.count(';'))


##### production_companies, production_countries
dict_to_col('production_companies', 'name')
dict_to_col('production_countries', 'iso_3166_1')

# filling the missing production countries
def get_country(row):
    a = row[2:]
    pc = ia.get_movie(a)['production countries']
    return pc

def to_iso(row):
    temp = ''
    for c in row:
        temp += mydict[c] + ';'
    return temp


# fill the NAs in production_countries
idx = df_cat[df_cat.production_countries.isnull()].index
df_cat.production_countries[idx] = df_cat.imdb_id[idx].map(lambda row: get_country(row))

# convert the country name as iso_code
iso_code = pd.read_csv('https://raw.githubusercontent.com/ybayle/ISRC/master/wikipedia-iso-country-codes.csv')
mydict = dict(zip(iso_code.iloc[:, 0], iso_code.iloc[:, 1]))
mydict['Russia'] = 'RU'
mydict['South Korea'] = 'KR'

df_cat.production_countries[idx] = df_cat.production_countries[idx].map(lambda row: to_iso(row))

df_cat['n_prod_comp'] = df_cat.production_companies.apply(lambda row: row.count(';'))
df_cat['n_prod_count'] = df_cat.production_countries.apply(lambda row: row.count(';'))


##### original_language, spoken_languages
dict_to_col('spoken_languages', 'iso_639_1')

# filling the missing values in n_spoken_languages
df_cat.loc[:, 'original_language'][df_cat.spoken_languages.isnull()]

idx = df_cat[df_cat.spoken_languages.isnull()].index
df_cat.spoken_languages[idx] = df_cat.original_language[idx]
df_cat['n_spoken_lang'] = df_cat.spoken_languages.apply(lambda row: row.count(';'))


##### cast
dict_to_col('cast', 'name')

df_cat.cast[df_cat.cast.isnull()] = 'No Info'
df_cat['n_cast'] = df_cat.cast.apply(lambda row: row.count(';'))


##### crew
# Fill NAs
df_cat['crew'] = df_cat['crew'].fillna('[{}]')

# Get the list of crew jobs
df_cat['crew_job'] = df_cat.crew.apply(lambda row: list_to_dict(row, 'job'))

# Get the number of elements
df_cat['n_crew'] = df_cat.crew_job.apply(lambda row: row.count(';'))

# Get the number of cast types
def job_counter(row):
    job_counts = Counter(row.split(';')[:-1])
    return len(job_counts)

df_cat['n_crew_job'] = df_cat.crew_job.apply(lambda row: job_counter(row))

#### DO MORE ANALYSIS HERE ####


##### release_date
df_cat.release_date[3828] = '10/12/01'

def fix_year(row):
    a = int(row.split('/')[2])
    if a <= 19:
        a += 2000
    else:
        a += 1900
    return a

df_cat['year'] = df_cat.release_date.map(lambda row: fix_year(row))
df_cat['month'] = df_cat.release_date.map(lambda row: row.split('/')[0])
df_cat['day'] = df_cat.release_date.map(lambda row: row.split('/')[1])

df_cat['release_date'] = pd.to_datetime({'year': df_cat.year,
                                        'month': df_cat.month,
                                        'day': df_cat.day})

df_cat['weekofday'] = df_cat.release_date.dt.dayofweek
df_cat['quarter'] = df_cat.release_date.dt.quarter



##### Keywords
dict_to_col('Keywords', 'name')

# Separate text columns
text_feats = ['Keywords', 'title', 'original_title', 'overview', 'tagline']
df_text = df_cat.loc[:, text_feats]

df_text.to_csv('data/tr_te_text.csv', encoding = 'utf-8-sig', index = False)


# drop the features
drop_feats = ['belongs_to_collection', 'imdb_id', 'homepage', 'poster_path', 'cast', 'crew']
df_cat.drop(drop_feats, axis = 1, inplace=True)
df_cat.drop(text_feats, axis = 1, inplace=True)

# Save the file
tr_cat = df_cat[:cut]
te_cat = df_cat[cut:]


tr_cat.to_csv('data/train_cat.csv', encoding = 'utf-8-sig', index = False)
te_cat.to_csv('data/test_cat.csv', encoding = 'utf-8-sig', index = False)
