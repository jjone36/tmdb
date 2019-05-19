import pandas as pd
import numpy as np

from sklearn.preprocessing import scale

# Import the all dataset
tr_num = pd.read_csv('data/train_num_p.csv')
tr_cat = pd.read_csv('data/train_cat_p.csv')

te_num = pd.read_csv('data/test_num_p.csv')
te_cat = pd.read_csv('data/test_cat_p.csv')


y = tr_num.revenue_log
tr_num.drop('revenue_log', axis = 1, inplace = True)

cut = len(tr_num)
df_num = pd.concat([tr_num, te_num], axis = 0)
df_cat = pd.concat([tr_cat, te_cat], axis = 0)

df = pd.concat([df_num, df_cat], axis = 1)

del tr_num, tr_cat, te_num, te_cat

############################################
# Numberic features
# Scaling
df['runtime_m'] = scale(df.runtime_m)

# n_cast, n_crew, n_crew_job
df['n_cast_log'] = np.log1p(df.n_cast)
df['n_crew_log'] = np.log1p(df.n_crew)
#df['n_crew_job_log'] = np.log1p(df.n_crew_job)    -> remove n_crew_job

# cast, crew genders
df['cast_male'] /= df.n_cast
df['crew_male'] /= df.n_crew

df.n_crew_profile[df.n_crew_profile.isnull()] = 0
df['n_crew_profile'] /= df.n_crew


# Mean Encoding
# genres
#df['md_ngenres_budget'] = df.groupby('n_genres')['budget'].transform('median')
df['r_popularity_ngenres'] = df.popularity / df.n_genres

# year
df['year_diff'] = df.year - np.min(df.year) + 1

#df['md_year_budget'] = df.groupby('year')['budget'].transform('median')
df['m_year_budget'] = df.groupby('year')['budget'].transform('min')

df['m_year_popularity'] = df.groupby('year')['popularity'].transform('mean')
df['m_year_runtime'] = df.groupby('year')['runtime'].transform('mean')
df['md_year_n_crew_log'] = df.groupby('year')['n_crew_log'].transform('median')
df['md_year_n_cast_log'] = df.groupby('year')['n_cast_log'].transform('median')

#df['r_popularity_year'] = df.popularity / df.year_diff
df['r_budget_year'] = df.budget / df.year_diff
#df['r_budget_year2'] = df.budget / (df.year_diff//10 + 1)

# month / weekofday
df['m_weekofday_budget'] = df.groupby('weekofday')['budget'].transform('mean')

# budget
#df['budget_ift'] = df.budget + df.budget*1.8/100*(2018 - df.year)
df['r_budget_runtime'] = df.budget / (df.runtime + 1)
#df['r_budget_popularity'] = df.budget / df.popularity

# produce_country & produce_company
#df['m_Ncnt_budget'] = df.groupby('n_prod_count')['budget'].transform('min')
#df['r_popularity_Ncnt'] = df.popularity / (df.n_prod_count + 1)
#df['md_Ncmp_budget'] = df.groupby('n_prod_comp')['budget'].transform('median')
#df['r_popularity_Ncmp'] = df.popularity / (df.n_prod_comp + 1)

# rating
voting = pd.read_csv("data/voting.csv")

df = pd.merge(df, voting, how = 'left', on = 'imdb_id')
df.reset_index(drop = True, inplace= True)

df['m_year_totalVotes'] = df.groupby("year")["totalVotes"].transform('mean')
df['m_rating_totalVotes'] = df.groupby("rating")["totalVotes"].transform('mean')

df['r_popularity_totalVotes'] = df['totalVotes']/df['popularity']
df['r_budget_totalVotes'] = df['budget']/df['totalVotes']

#df['r_totalVotes_rating'] = df['totalVotes']/df['rating']
df['r_budget_rating'] = df['budget']/df['rating']
df['r_runtime_rating'] = df['runtime']/df['rating']

df['r_totalVotes_year'] = df['totalVotes']/df['year']
df['r_rating_popularity'] = df['rating']/df['popularity']

###################
# text data
df_text = pd.read_csv('data/tr_te_text.csv')

# keywords
df_text.Keywords[df_text.Keywords.isnull()] = ''
df_text['n_Keywords'] = df_text.Keywords.apply(lambda row: row.count(';'))

# title & original_title
df_text.title[df_text.title.isnull()] = ''
df_text['title_equal'] = (df_text.title == df_text.original_title)*1
#df_text['title_nwords'] = df_text.title.apply(lambda row: len(row.split()))
df_text['title_len'] = df_text.title.map(len)

# overview
df_text.overview[df_text.overview.isnull()] = ''
df_text['overview_len'] = df_text.overview.map(len)
#df_text['overview_nwords'] = df_text.overview.apply(lambda row: len(row.split()))

# tagline
df_text.tagline[df_text.tagline.isnull()] = ''
df_text['tagline_len'] = df_text.tagline.map(len)
#df_text['tagline_nwords'] = df_text.tagline.apply(lambda row: len(row.split()))

df_text = df_text.iloc[:, 5:]
#df_text.reset_index(drop = True, inplace = True)


# Combine all the data
df.reset_index(drop = True, inplace = True)
df_all = pd.concat([df, df_text], axis = 1)

# Seperate the dataframe
tr = df[:cut]
tr['revenue_log'] = y

te = df[cut:]

# Save the files
tr.to_csv('data/train_2.csv', index = False)
te.to_csv('data/test_2.csv', index = False)
