# Code to generate train-test split data from learners.py

import feather
import pandas
from sklearn.model_selection import train_test_split

def gen_data(data_fpath, test_size=0.2, random_state=1234, features_to_drop=[]):
  wine = feather.read_dataframe(data_fpath)
  x = wine.drop(['quality'] + features_to_drop, axis=1)
  y = wine.quality
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
  return x_train, x_test, y_train, y_test
data_fpath = './intermediate/wine_logged_scaled.feather'
x_train, x_test, y_train, y_test = gen_data(data_fpath)

