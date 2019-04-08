import pandas as pd
import numpy as np

import ast
from datetime import date

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

# Cleaning- Budget, Revenue in train set
tr.loc[tr['id'] == 16,'revenue'] = 192864          # Skinning
tr.loc[tr['id'] == 90,'budget'] = 30000000         # Sommersby
tr.loc[tr['id'] == 118,'budget'] = 60000000        # Wild Hogs
tr.loc[tr['id'] == 149,'budget'] = 18000000        # Beethoven
tr.loc[tr['id'] == 313,'revenue'] = 12000000       # The Cookout
tr.loc[tr['id'] == 451,'revenue'] = 12000000       # Chasing Liberty
tr.loc[tr['id'] == 464,'budget'] = 20000000        # Parenthood
tr.loc[tr['id'] == 470,'budget'] = 13000000        # The Karate Kid, Part II
tr.loc[tr['id'] == 513,'budget'] = 930000          # From Prada to Nada
tr.loc[tr['id'] == 797,'budget'] = 8000000         # Welcome to Dongmakgol
tr.loc[tr['id'] == 819,'budget'] = 90000000        # Alvin and the Chipmunks: The Road Chip
tr.loc[tr['id'] == 850,'budget'] = 90000000        # Modern Times
tr.loc[tr['id'] == 1007,'budget'] = 2              # Zyzzyx Road
tr.loc[tr['id'] == 1112,'budget'] = 7500000        # An Officer and a Gentleman
tr.loc[tr['id'] == 1131,'budget'] = 4300000        # Smokey and the Bandit
tr.loc[tr['id'] == 1359,'budget'] = 10000000       # Stir Crazy
tr.loc[tr['id'] == 1542,'budget'] = 1              # All at Once
tr.loc[tr['id'] == 1570,'budget'] = 15800000       # Crocodile Dundee II
tr.loc[tr['id'] == 1571,'budget'] = 4000000        # Lady and the Tramp
tr.loc[tr['id'] == 1714,'budget'] = 46000000       # The Recruit
tr.loc[tr['id'] == 1721,'budget'] = 17500000       # Cocoon
tr.loc[tr['id'] == 1865,'revenue'] = 25000000      # Scooby-Doo 2: Monsters Unleashed
tr.loc[tr['id'] == 1885,'budget'] = 12             # In the Cut
tr.loc[tr['id'] == 2091,'budget'] = 10             # Deadfall
tr.loc[tr['id'] == 2268,'budget'] = 17500000       # Madea Goes to Jail budget
tr.loc[tr['id'] == 2491,'budget'] = 6              # Never Talk to Strangers
tr.loc[tr['id'] == 2602,'budget'] = 31000000       # Mr. Holland's Opus
tr.loc[tr['id'] == 2612,'budget'] = 15000000       # Field of Dreams
tr.loc[tr['id'] == 2696,'budget'] = 10000000       # Nurse 3-D
tr.loc[tr['id'] == 2801,'budget'] = 10000000       # Fracture
tr.loc[tr['id'] == 335,'budget'] = 2
tr.loc[tr['id'] == 348,'budget'] = 12
tr.loc[tr['id'] == 470,'budget'] = 13000000
tr.loc[tr['id'] == 513,'budget'] = 1100000
tr.loc[tr['id'] == 640,'budget'] = 6
tr.loc[tr['id'] == 696,'budget'] = 1
tr.loc[tr['id'] == 797,'budget'] = 8000000
tr.loc[tr['id'] == 850,'budget'] = 1500000
tr.loc[tr['id'] == 1199,'budget'] = 5
tr.loc[tr['id'] == 1282,'budget'] = 9              # Death at a Funeral
tr.loc[tr['id'] == 1347,'budget'] = 1
tr.loc[tr['id'] == 1755,'budget'] = 2
tr.loc[tr['id'] == 1801,'budget'] = 5
tr.loc[tr['id'] == 1918,'budget'] = 592
tr.loc[tr['id'] == 2033,'budget'] = 4
tr.loc[tr['id'] == 2118,'budget'] = 344
tr.loc[tr['id'] == 2252,'budget'] = 130
tr.loc[tr['id'] == 2256,'budget'] = 1
tr.loc[tr['id'] == 2696,'budget'] = 10000000

# Cleaning- Budget, Revenue in test set
te.loc[te['id'] == 6733,'budget'] = 5000000
te.loc[te['id'] == 3889,'budget'] = 15000000
te.loc[te['id'] == 6683,'budget'] = 50000000
te.loc[te['id'] == 5704,'budget'] = 4300000
te.loc[te['id'] == 6109,'budget'] = 281756
te.loc[te['id'] == 7242,'budget'] = 10000000
te.loc[te['id'] == 7021,'budget'] = 17540562       #  Two Is a Family
te.loc[te['id'] == 5591,'budget'] = 4000000        # The Orphanage
te.loc[te['id'] == 4282,'budget'] = 20000000       # Big Top Pee-wee
te.loc[te['id'] == 3033,'budget'] = 250
te.loc[te['id'] == 3051,'budget'] = 50
te.loc[te['id'] == 3084,'budget'] = 337
te.loc[te['id'] == 3224,'budget'] = 4
te.loc[te['id'] == 3594,'budget'] = 25
te.loc[te['id'] == 3619,'budget'] = 500
te.loc[te['id'] == 3831,'budget'] = 3
te.loc[te['id'] == 3935,'budget'] = 500
te.loc[te['id'] == 4049,'budget'] = 995946
te.loc[te['id'] == 4424,'budget'] = 3
te.loc[te['id'] == 4460,'budget'] = 8
te.loc[te['id'] == 4555,'budget'] = 1200000
te.loc[te['id'] == 4624,'budget'] = 30
te.loc[te['id'] == 4645,'budget'] = 500
te.loc[te['id'] == 4709,'budget'] = 450
te.loc[te['id'] == 4839,'budget'] = 7
te.loc[te['id'] == 3125,'budget'] = 25
te.loc[te['id'] == 3142,'budget'] = 1
te.loc[te['id'] == 3201,'budget'] = 450
te.loc[te['id'] == 3222,'budget'] = 6
te.loc[te['id'] == 3545,'budget'] = 38
te.loc[te['id'] == 3670,'budget'] = 18
te.loc[te['id'] == 3792,'budget'] = 19
te.loc[te['id'] == 3881,'budget'] = 7
te.loc[te['id'] == 3969,'budget'] = 400
te.loc[te['id'] == 4196,'budget'] = 6
te.loc[te['id'] == 4221,'budget'] = 11
te.loc[te['id'] == 4222,'budget'] = 500
te.loc[te['id'] == 4285,'budget'] = 11
te.loc[te['id'] == 4319,'budget'] = 1
te.loc[te['id'] == 4639,'budget'] = 10
te.loc[te['id'] == 4719,'budget'] = 45
te.loc[te['id'] == 4822,'budget'] = 22
te.loc[te['id'] == 4829,'budget'] = 20
te.loc[te['id'] == 4969,'budget'] = 20
te.loc[te['id'] == 5021,'budget'] = 40
te.loc[te['id'] == 5035,'budget'] = 1
te.loc[te['id'] == 5063,'budget'] = 14
te.loc[te['id'] == 5119,'budget'] = 2
te.loc[te['id'] == 5214,'budget'] = 30
te.loc[te['id'] == 5221,'budget'] = 50
te.loc[te['id'] == 4903,'budget'] = 15
te.loc[te['id'] == 4983,'budget'] = 3
te.loc[te['id'] == 5102,'budget'] = 28
te.loc[te['id'] == 5217,'budget'] = 75
te.loc[te['id'] == 5224,'budget'] = 3
te.loc[te['id'] == 5469,'budget'] = 20
te.loc[te['id'] == 5840,'budget'] = 1
te.loc[te['id'] == 5960,'budget'] = 30
te.loc[te['id'] == 6506,'budget'] = 11
te.loc[te['id'] == 6553,'budget'] = 280
te.loc[te['id'] == 6561,'budget'] = 7
te.loc[te['id'] == 6582,'budget'] = 218
te.loc[te['id'] == 6638,'budget'] = 5
te.loc[te['id'] == 6749,'budget'] = 8
te.loc[te['id'] == 6759,'budget'] = 50
te.loc[te['id'] == 6856,'budget'] = 10
te.loc[te['id'] == 6858,'budget'] =  100
te.loc[te['id'] == 6876,'budget'] =  250
te.loc[te['id'] == 6972,'budget'] = 1
te.loc[te['id'] == 7079,'budget'] = 8000000
te.loc[te['id'] == 7150,'budget'] = 118
te.loc[te['id'] == 6506,'budget'] = 118
te.loc[te['id'] == 7225,'budget'] = 6
te.loc[te['id'] == 7231,'budget'] = 85
te.loc[te['id'] == 5222,'budget'] = 5
te.loc[te['id'] == 5322,'budget'] = 90
te.loc[te['id'] == 5350,'budget'] = 70
te.loc[te['id'] == 5378,'budget'] = 10
te.loc[te['id'] == 5545,'budget'] = 80
te.loc[te['id'] == 5810,'budget'] = 8
te.loc[te['id'] == 5926,'budget'] = 300
te.loc[te['id'] == 5927,'budget'] = 4
te.loc[te['id'] == 5986,'budget'] = 1
te.loc[te['id'] == 6053,'budget'] = 20
te.loc[te['id'] == 6104,'budget'] = 1
te.loc[te['id'] == 6130,'budget'] = 30
te.loc[te['id'] == 6301,'budget'] = 150
te.loc[te['id'] == 6276,'budget'] = 100
te.loc[te['id'] == 6473,'budget'] = 100
te.loc[te['id'] == 6842,'budget'] = 30


# Separate categorical & numerical features
tr_cat = tr[cat_feats]
tr_num = tr[num_feats]

te_cat = te[cat_feats]
te_num = te[num_feats[:-1]]

# Preprocessing - numeric feature
cut = len(tr)
df_num = pd.concat([tr_num, te_num], axis = 0).reset_index(drop = True)

# runtime
df_num['runtime_h'], df_num['runtime_m'] = df_num.runtime // 60, df_num.runtime % 60

# scaling
df_num['revenue_log'] = np.log1p(df_num.revenue)
df_num['runtime_log']= np.log1p(tr_num.runtime)
df_num['budget_log'] = np.log1p(df_num.budget)
df_num['popularity_log']= np.log1p(df_num.popularity)


# Preprocessing - categorical feature
# Concat train and test set
df_cat = pd.concat([tr_cat, te_cat], axis = 0).reset_index(drop = True)

# belongs to collection
df_cat['is_collection'] = df_cat.belongs_to_collection.notnull()

# release_date
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
    # The number of elements
    df_cat['n_'+col] = df_cat[col].apply(lambda row: len(row.split(';')) - 1)


# genres
dict_to_col('genres', 'name')

df_cat.loc[:, ['title', 'release_date']][df_cat.genres == '']
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


# spoken_languages
dict_to_col('spoken_languages', 'iso_639_1')

# production_companies, production_countries
dict_to_col('production_companies', 'name')
dict_to_col('production_countries', 'iso_3166_1')

# cast
dict_to_col('cast', 'name')

# crew
# Fill NAs
df_cat['crew'] = df_cat['crew'].fillna('[{}]')
# Get the list of crew jobs
df_cat['crew_job'] = df_cat.crew.apply(lambda row: list_to_dict(row, 'job'))
# Get the number of elements
df_cat['n_crew_job'] = df_cat.crew_job.apply(lambda row: len(row.split(';')) -1)

#### DO MORE ANALYSIS HERE ####

# Keywords
dict_to_col('Keywords', 'name')
# Separate text columns
text_feats = ['Keywords', 'original_title', 'overview', 'tagline']
df_text = df_cat.loc[:, text_feats]

#### DO MORE ANALYSIS HERE ####


drop_feats = ['belongs_to_collection', 'homepage', 'poster_path', 'cast', 'crew']
df_cat.drop(drop_feats, axis = 1, inplace=True)
df_cat.drop(text_feats, axis = 1, inplace=True)


# Save the file
tr_num = df_num[:cut]
te_num = df_num[cut:]

tr_cat = df_cat[:cut]
te_cat = df_cat[cut:]

tr = pd.concat([tr_num, tr_cat], axis = 1)
te = pd.concat([te_num, te_cat], axis = 1)

tr.to_csv('data/tr_processed.csv', encoding = 'utf-8-sig', index = False)
te.to_csv('data/te_processed.csv', encoding = 'utf-8-sig', index = False)
