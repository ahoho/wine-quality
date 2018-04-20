# Title: Code for making ridgeplots only
# Author: Meenakshi Parameshwaran
# Date: 20/04/18

# An older version of the code #

# Load packages
library(pacman)
p_load(tidyverse, ggridges, magrittr)

# read in the original csv files
red <- read.csv(file = "./input/winequality-red.csv", header = T, sep = ";")
white <- read.csv(file = "./input/winequality/winequality-white.csv", header = T, sep = ";")
  
# add a dummy that is 1 for red wine and 0 for white wine (number of obs: red wine - 1599; white wine - 4898)
red$winecolour <- 1
white$winecolour <- 0

# check the columns are the same in both dfs
names(red) == names(white)
    
# stack the two dataframes to make a single wine df
wine <- rbind(red, white)

# check the result
dim(wine)
str(wine)
head(wine)
summary(wine)
      
# make the quality variable a factor? - median value is 6
wine$quality <- as.factor(wine$quality)

# make the colour variable a factor
wine$winecolour <- factor(wine$winecolour, labels = c("white", "red"))

# remove the red and white dfs, just use the joint one now
rm(red, white)

# distribution of wine quality by colour
with(wine, summarytools::ctable(winecolour, quality))
# ok colours are coded correctly at this point
          
# use the untransformed data for the plots
origwine <- wine

# make all columns numeric again
origwine %<>% mutate_if(is.character, as.numeric)

# ridge plot function
ridgeplot <- function(winevar) {
    origwine %>%
      ggplot(aes(y = quality)) +
      geom_density_ridges(aes_q(x = as.name(winevar), 
                                fill = paste(origwine$quality, origwine$winecolour)), 
                          alpha = .8, 
                          color = "white", 
                          rel_min_height = 0.01) +
    labs(x = Hmisc::capitalize(gsub(x = winevar,
                                    pattern = "\\.",
                                    replacement = " ")),
         y = "Wine quality") +
    scale_y_discrete(expand = c(0.01, 0)) +
    scale_x_continuous(expand = c(0.01, 0)) +
    scale_fill_cyclical(breaks = c("3 red", "3 white"),
                        labels = c(`3 red` = "Red", `3 white` = "White"),
                        values = c("3 red" = "#E31A1C", "3 white" = "#33A02C", "4 red" = "#FB9A99", "4 white" = "#B2DF8A", "5 red" = "#E31A1C", "5 white" = "#33A02C", "6 red" = "#FB9A99", "6 white" = "#B2DF8A", "7 red" = "#E31A1C", "7 white" = "#33A02C", "8 red" = "#FB9A99", "8 white" = "#B2DF8A", "9 white" = "#33A02C"),
                        name = "Wine colour", 
                        guide = "legend") +
    theme_ridges(grid = FALSE, font_size = 11)
}

# make ridgeplots
ridgeplotlist <- lapply(names(origwine)[1:11], ridgeplot)
ridgeplots_grid <- ggpubr::ggarrange(plotlist = ridgeplotlist, ncol = 2, nrow = 2)

# write plots to PDF
# if(!file.exists("winevars_ridgeplots.pdf")){
#       ggpubr::ggexport(ridgeplots_grid, filename = "winevars_ridgeplots.pdf")
#   }
