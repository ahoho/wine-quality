# Script to format benchmarking results tables #

library(pacman)
p_load(tidyverse, broom, knitr, stringr, here, purrr, kableExtra)


# Function to make formatted benchmark results table
benchmarktable <- function(file, task, measure, out = "latex") {

df <- read.csv(file = here("output", file))
capvars <- paste(str_split(str_remove(file, ".csv"), "_", simplify = TRUE)[3:2], collapse = " ")

df %>% 
  filter(type == task) %>% 
  filter(metric %in% measure) %>% 
  select(-type) %>%
  mutate(metric = case_when(
    metric == "f1_score" ~ str_replace(str_to_title(metric), "_", " "),
    TRUE ~ str_to_upper(metric)
    )) %>% 
  mutate(model = case_when(
    model %in% c("dummy","linear") ~ str_to_title(model),
    TRUE ~ str_to_upper(model)
    )) %>% 
  rename_at(., vars(starts_with("m")), str_to_title) %>% 
  rename_at(., vars(ends_with("b")), str_to_upper) %>% 
  kable(digits = 4,
        format = out,
        caption = paste("Benchmarking results -", ifelse(task == "reg", "regression", "classification"), "task with", capvars, "variables")) %>% 
  kableExtra::row_spec(0, bold = TRUE)

}

# Make the benchmarking tables
files <- c("results_unscaled_all.csv", "results_chemical_all.csv", "results_color_all.csv")
tasks <- c("reg", "clf")
measures <- c("mse", "mae", "f1_score")

benchdata <- list(files = files, tasks = tasks, measures = measures) 

benchtablelist <- 
  benchdata %>%
    purrr::cross_df() %>% 
    filter(!(tasks == "reg" & measures == "f1_score")) %>% 
    pmap(., ~ benchmarktable(..1, ..2, ..3))


# Function to make formatted benchmarking plots

benchmarkplot <- function(file, task, measure) {
  
  df <- read.csv(file = here("output", file))
  capvars <- paste(str_split(str_remove(file, ".csv"), "_", simplify = TRUE)[3:2], collapse = " ")
  fmeasure <- ifelse(measure == "f1_score", str_replace(str_to_title(measure), "_", " "), str_to_upper(measure))
  
  df %>% 
    filter(type == task) %>% 
    filter(metric %in% measure) %>% 
    select(-type) %>%
    mutate(metric = case_when(
      metric == "f1_score" ~ str_replace(str_to_title(metric), "_", " "),
      TRUE ~ str_to_upper(metric)
    )) %>% 
    mutate(model = case_when(
      model %in% c("dummy","linear") ~ str_to_title(model),
      TRUE ~ str_to_upper(model)
    )) %>% 
    rename_at(., vars(starts_with("m")), str_to_title) %>% 
    rename_at(., vars(ends_with("b")), str_to_upper) %>% 
    ggplot(aes(Mean, Model, color = Model)) +
    geom_point() +
    geom_errorbarh(aes(xmin = LB, xmax = UB)) +
    theme(legend.position = "none") +
    xlab(paste("Mean", fmeasure)) +
    ggtitle(paste("Benchmarking results -", ifelse(task == "reg", "regression", "classification"), "task with", capvars, "variables"))
  
}

# Make the benchmarking plots
files <- c("results_unscaled_all.csv", "results_chemical_all.csv", "results_color_all.csv")
tasks <- c("reg", "clf")
measures <- c("mse", "mae", "f1_score")

benchdata <- list(files = files, tasks = tasks, measures = measures) 

benchplotlist <- 
  benchdata %>%
  purrr::cross_df() %>% 
  filter(!(tasks == "reg" & measures == "f1_score")) %>% 
  pmap(., ~ benchmarkplot(..1, ..2, ..3))
