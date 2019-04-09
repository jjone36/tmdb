import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from Keras.layers import Dense

# Import the all dataset
tr = pd.read_csv('data/train_2.csv')
te = pd.read_csv('data/test_2.csv')

# Additional preprocessing (if neccesssary)

# Split into train and Valid set
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size = .2, random_state = 36)


# Modeling
model = Sequential()


# Evaluation
