import pickle

import feather
import scipy
import numpy as np

from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier

from sklearn.model_selection import (
  train_test_split, RandomizedSearchCV, KFold, StratifiedKFold)
from sklearn.metrics import (
  mean_squared_error, mean_absolute_error, f1_score, make_scorer)
from sklearn.utils import resample

def gen_data(data_fpath, test_size=0.2, random_state=1234, features_to_drop=[]):
  """
  Generate train and test data, optionally dropping features
  """
  # read in data
  wine = feather.read_dataframe(data_fpath)

  # split into x and y data
  x = wine.drop(['quality'] + features_to_drop, axis=1)
  y = wine.quality

  # split into train and test
  x_train, x_test, y_train, y_test = train_test_split(
      x, y, test_size=test_size, random_state=random_state)

  return x_train, x_test, y_train, y_test

def config_cv(learner, scoring=None, params=None,
              n_iter=300, folds=3, n_jobs=8, **kwargs):
    """
    Pre-configure the RandomizedSearchCV
    """
    if isinstance(learner, (LogisticRegression, SVC, RandomForestClassifier)):
      params = {**params, 'class_weight': ['balanced', None]}
    if isinstance(learner, LinearRegression):
      params = {}
    if not params:
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

def train(model_dir, x_train, y_train):
  """
  Train models
  """
  obs, features = x_train.shape

  # global configurations
  mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
  mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
  f1_scorer = make_scorer(f1_score, labels=[3,4,5,6,7,8,9], average='micro')

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

  scorers_reg = {'mse': mse_scorer, 'mae': mae_scorer}
  regressors = {
    name: config_cv(learner, scorers_reg, p, refit='mse') 
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

  scorers_clf = {'mse': mse_scorer, 'mae': mae_scorer, 'f1': f1_scorer}
  classifiers = {
    name: config_cv(learner, scorers_clf, p, refit='f1') 
    for learner, (name, p) in zip(classifiers, params)
  }

  # perform training
  np.random.seed(12345)
  for name, reg in regressors.items():
      print('On {} regressor'.format(name))
      reg.fit(x_train, y_train)
      with open('./intermediate/{}/reg_{}.pkl'.format(model_dir, name), 'wb') as o:
          pickle.dump(reg, o)

  np.random.seed(12345)
  for name, clf in classifiers. items():
      print('On {} classifier'.format(name))
      clf.fit(x_train, y_train)
      with open('./intermediate/{}/clf_{}.pkl'.format(model_dir, name), 'wb') as o:
          pickle.dump(clf, o)

  def load_models(model_dir):
    """
    Load saved models
    """
    model_names = ['dummy', 'linear', 'svm', 'rf', 'mlp']
    regressors, classifiers = {}, {}

    for name in model_names:
      with open('./intermediate/{}/reg_{}.pkl'.format(model_dir, name), 'rb') as i:
        regressors[name] = pickle.load(i)
      with open('./intermediate/{}/clf_{}.pkl'.format(model_dir, name), 'rb') as i:
        classifiers[name] = pickle.load(i)

    return regressors, classifiers

  def test_err_cv(model, metric, test_x, test_y, n_splits=3, seed=1234):
    """
    Calculate mean and confidence intervals for a metric on a cross-validated
    dataset
    N.B.: metric should be point-wise
    """
    # unsure whether to use stratified k-fold or regular k-fold?
    skf = StratifiedKFold(n_splits=3)
    errors = np.array([])

    for idx, _ in skf.split(test_x, test_y):
      test_x_fold, test_y_fold = test_x[idx], text_y[idx]
      pred_y_fold = model.predict(test_x_fold)
      error = metric(test_y_fold, pred_y_fold)
      errors = np.concatenate([errors, error])

    mean = np.mean(errors)
    # TODO:


  def test_err_bootstrap(model, metric, x_test, y_test,
                         n_iter=1000, alpha=0.05, seed=1234):
    """
    Calculate mean and 100*(1-alpha)% central confidence intervals for a metric
    on a bootstrapped dataset
    """
    np.random.seed(seed)
    results = []

    for _ in range(n_iter):
      # by default, samples the same number in array, with replacement
      x_test_bs, y_test_bs = resample(x_test, y_test)
      y_pred_bs = model.predict(x_test_bs)
      results.append(metric(y_test_bs, y_pred_bs))
    
    a = 100 * alpha / 2
    lb, ub = np.percentile(results, (a, 100 - a))
    mean = np.mean(results)

    return mean, lb, ub

if __name__ == '__main__':
  data_fpath = './intermediate/wine_logged_unscaled.feather'
  x_train, x_test, y_train, y_test = gen_data(data_fpath)
  # train on all data
  train('unscaled')

  # TODO:
  # finish variance reduction/kfold
  # 
