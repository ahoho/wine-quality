# Fitting a linear model to the wine quality train data

# This script loads the scaled and logged dataset, performs the same train-test split as in learners.py, then fits simple linear and logistic regression models. It outputs formatted regression tables in latex.

# load packages
library(pacman)
p_load(tidyverse, here, broom, knitr, reticulate)

# use python 3
pypath = "/Applications/anaconda/bin/python3.6"
condapath = "/Applications/anaconda/bin/conda"
conda_py3env = "py-env"

#use_python(python = pypath)
use_condaenv(condaenv = conda_py3env , conda = condapath)

# set seed
py_set_seed(1234)
# run the data generating code
py_run_file("gen_data_train_test.py")


# now back to R - the train and test objects exist in R using py$

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
      term != "pH" ~ stringr::str_to_title(term),
      TRUE ~ "pH"
    )) %>% 
    rename(Variable = term, 
           Estimate = estimate, 
           SE = std.error, 
           Statistic = statistic) %>% 
    kable(caption = "Linear model - logged and scaled training data", 
          format = "latex",
          digits = 3)

# for interest: run a linear model by wine colour and output to kable
# wine %>% 
#   group_by(color) %>%
#   do(tidy(lm(quality ~ ., data = .))) %>% 
#   mutate(term = stringr::str_replace_all(term, "`", "")) %>% 
#   mutate(term = case_when(
#     term == "(Intercept)" ~ "Intercept",
#     term != "pH" ~ Hmisc::capitalize(term),
#     TRUE ~ "pH"
#   )) %>% 
#   rename(Variable = term, 
#          Estimate = estimate, 
#          SE = std.error, 
#          Statistic = statistic) %>% 
#   kable(caption = "Linear model - all training data", 
#         format = "latex",
#         digits = 3)

# for interest: run a logit model  and output to kable
# wine %>% 
#   do(tidy(nnet::multinom(quality ~ ., data = .))) %>% 
#   mutate(term = stringr::str_replace_all(term, "`", "")) %>% 
#   mutate(term = case_when(
#     term == "(Intercept)" ~ "Intercept",
#     term != "pH" ~ Hmisc::capitalize(term),
#     TRUE ~ "pH"
#   )) %>% 
#   rename(Variable = term, 
#          Estimate = estimate, 
#          SE = std.error, 
#          Statistic = statistic) %>% 
#   kable(caption = "Linear model - all training data", 
#         format = "latex",
#         digits = 3)