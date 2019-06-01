# TMDB Box Office Prediction
: *Kaggle competition predicting movie revenues*

<br>

## ***1. What movies make the most profit in the film industry?***

My solution for the Kaggle competition, [TMDB Box Office Prediction](https://www.kaggle.com/c/tmdb-box-office-prediction). The competition is predicting the box office revenues with the metadata of over 7,000 past films from [TMDB Movie Database](https://www.themoviedb.org/). The dataset contains various information such as title, original language and spoken languages, release year and running time, the list of crew and cast members, the production country and company, keywords and tagline.

![page](https://github.com/jjone36/tmdb/blob/master/img.PNG)

The metric used is the root-mean-squared error and the accuracy of the first baseline model is **2.1347** by a Elastic Net model. The final submission public LB score **1.7249**, Top 18% (242nd of 1400) and private LB score **1.----**. The best single model I've built during the competition was a CatBoost model (max_depth = 9, learning_rate = .05). The final prediction has made by stacking 3 layers with residual weighted boosting technique and ensembling.  

<br>

* **Project Date:** Apr - May, 2019
* **Applied skills:** Data Preprocessing and Manipulation, Scraping data with TMDB API, Intensive Exploratory Data Analysis, Feature engineering, Cross Validation, Residual Weighted Boosting, Stacking Models, and Ensemble Learning.  

<br>

## ***2. FlowChart***

The whole process is as shown below.   

![page](https://github.com/jjone36/tmdb/blob/master/flow.PNG)

<br>

## ***3. File Details***
- **01.preprocessing**: [Data overviewing](https://github.com/jjone36/Cosmetic/blob/master/01.preprocessing/00_overview.py) and Preprocessing [numerical data](https://github.com/jjone36/Cosmetic/blob/master/01.preprocessing/01_1_preprocessing_num.py) and [categorical data](https://github.com/jjone36/Cosmetic/blob/master/01.preprocessing/01_1_preprocessing_cat.py).

- **02.EDA**: [Exploratory data analysis and Visualization](https://github.com/jjone36/Cosmetic/blob/master/02.eda/02_eda.ipynb).

- **03.Feature Engineering**: [Feature engineering](https://github.com/jjone36/Cosmetic/blob/master/03.feature_engineering/03_1_feature_engineering.py) and [Creating more features with label encoding](https://github.com/jjone36/Cosmetic/blob/master/03.feature_engineering/03_2_additional_features.py)   

- **04.Modeling**: The whole trial & error notebooks and [the final modeling](https://github.com/jjone36/Cosmetic/blob/master/04.modeling/04_modeling.py)

<br>
