# Title: Fitting a linear model to the wine quality train data
# Author: Meenkshi Parameshwaran
# Date: 20/04/18

# load packages
library(pacman)
p_load(tidyverse, here, broom, knitr, reticulate)

# use Python 3
# reticulate::use_condaenv(condaenv = "py-env", conda = "/Applications/anaconda/bin/conda")
reticulate::use_python("/Applications/anaconda/bin/python3.6")

# check feather is available
reticulate::py_module_available("feather")

# load the dataset and run the train test split - do this in interactive python as source_python("gen_data_train_test.py") isn't working

repl_python()

# this is the code to enter in interactive python

# import feather
# import pandas
# from sklearn.model_selection import train_test_split
# def gen_data(data_fpath, test_size=0.2, random_state=1234, features_to_drop=[]):
#   wine = feather.read_dataframe(data_fpath)
#   x = wine.drop(['quality'] + features_to_drop, axis=1)
#   y = wine.quality
#   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
#   return x_train, x_test, y_train, y_test
#   data_fpath = './intermediate/wine_logged_unscaled.feather'
#   x_train, x_test, y_train, y_test = gen_data(data_fpath)
#   exit

# now back to R - the objects exist in R using py$

# create the training set wine object by combining the x_train and y_train objects
wine <- cbind(py$x_train, py$y_train)
wine$quality <- wine$`py$y_train`
wine$`py$y_train` <- NULL

# run a linear model on mean-centered data and output to kable
wine %>% 
do(tidy(lm(quality ~ ., data = .))) %>% 
  mutate(term = stringr::str_replace_all(term, "`", "")) %>% 
  mutate(term = case_when(
    term == "(Intercept)" ~ "Intercept",
    term != "pH" ~ Hmisc::capitalize(term),
    TRUE ~ "pH"
  )) %>% 
  rename(Variable = term, 
         Estimate = estimate, 
         SE = std.error, 
         Statistic = statistic) %>% 
  kable(caption = "Linear model - all training data", 
        digits = 3)

# for interest: run a linear model by wine colour and output to kable
wine %>% 
  group_by(color) %>%
  do(tidy(lm(quality ~ ., data = .))) %>% 
  mutate(term = stringr::str_replace_all(term, "`", "")) %>% 
  mutate(term = case_when(
    term == "(Intercept)" ~ "Intercept",
    term != "pH" ~ Hmisc::capitalize(term),
    TRUE ~ "pH"
  )) %>% 
  rename(Variable = term, 
         Estimate = estimate, 
         SE = std.error, 
         Statistic = statistic) %>% 
  kable(caption = "Linear model - all training data", 
        digits = 3)

# for interest: run a logit model  and output to kable
wine %>% 
  do(tidy(nnet::multinom(quality ~ ., data = .))) %>% 
  mutate(term = stringr::str_replace_all(term, "`", "")) %>% 
  mutate(term = case_when(
    term == "(Intercept)" ~ "Intercept",
    term != "pH" ~ Hmisc::capitalize(term),
    TRUE ~ "pH"
  )) %>% 
  rename(Variable = term, 
         Estimate = estimate, 
         SE = std.error, 
         Statistic = statistic) %>% 
  kable(caption = "Linear model - all training data", 
        digits = 3)