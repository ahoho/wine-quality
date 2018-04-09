import pickle

import feather
import scipy
import numpy as np

from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, f1_score, make_scorer

# read in data
wine = feather.read_dataframe('./intermediate/wine_logged_scaled.feather')

# split into x and y data
x = wine.drop(['quality'], axis=1)
y = wine.quality

# split into train and test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1234)

obs, features = x_train.shape

# global configurations
reg_score = make_scorer(mean_squared_error, greater_is_better=False)

def config_reg_cv(learner, params=None, scoring=reg_score,
                  n_iter=200, folds=10, n_jobs=6, **kwargs):
  """
  Pre-configure the RandomizedSearchCV for a classifier
  """
  if isinstance(learner, (LinearRegression)):
    params = {}
  if len(params) == 0:
    n_iter = 1
  learner = RandomizedSearchCV(
    learner,
    params,
    scoring=scoring,
    n_iter=n_iter,
    cv=folds,
    n_jobs=n_jobs, 
    **kwargs
  )
  return learner
  
def config_clf_cv(learner, params=None, scoring=f1_score,
                  n_iter=200, folds=10, n_jobs=7, **kwargs):
  """
  Pre-configure the RandomizedSearchCV for a classifier
  """
  if not isinstance(learner, (DummyClassifier)):
    params = {**params, 'class_weight': ['balanced', None]}
  if len(params) == 0:
    n_iter = 1
  learner = RandomizedSearchCV(
    learner,
    params,
    scoring=scoring,
    n_iter=n_iter,
    cv=folds,
    n_jobs=n_jobs, 
    **kwargs
  )
  return learner

# build configurations for each of the learners
params = [
  ('dummy', {}),
  ('linear', {
      'penalty': ['l1', 'l2'],
      'C': scipy.stats.expon(scale=100)
    }
  ),  
  ('svm', {
      'C': scipy.stats.expon(scale=100),
      'gamma': scipy.stats.expon(scale=.1),
      'kernel': ['rbf']
    }
  ),
  ('rf', {
      'n_estimators': scipy.stats.randint(low=10, high=100),
      'max_features': scipy.stats.randint(low=2, high=features),
      'max_depth': [None] # consider making this random as well
    }
  ),
  ('mlp', {
      'activation': ['relu', 'logistic'],
      'hidden_layer_sizes': [(32, 32), (64, 64), (128, 128)],
      'learning_rate_init': scipy.stats.norm(loc=0.001, scale=0.0002),
      'max_iter': [500],
    }
  )
]

# create regressors
regressors = [
  DummyRegressor(),
  LinearRegression(),
  SVR(),
  RandomForestRegressor(),
  MLPRegressor()
]
regressors = {
  name: config_reg_cv(learner, p) 
  for learner, (name, p) in zip(regressors, params)
}

# create classifiers
classifiers = [
  DummyClassifier(),
  LogisticRegression(),
  SVC(),
  RandomForestClassifier(),
  MLPClassifier()
]

classifiers = {
  name: config_clf_cv(learner, p) 
  for learner, (name, p) in zip(classifiers, params)
}

# perform training
np.random.seed(12345)
for name, reg in regressors.items():
    print('On {} regressor'.format(name))
    if name in ['mlp']:
      reg.fit(x_train, y_train)
      with open('./intermediate/reg_{}.pkl'.format(name), 'wb') as o:
          pickle.dump(reg, o)

np.random.seed(12345)
for name, clf in classifiers.items():
    print('On {} classifier'.format(name))
    clf.fit(x_train, y_train)
    with open('./intermediate/clf_{}.pkl'.format(name), 'wb') as o:
        pickle.dump(clf, o)