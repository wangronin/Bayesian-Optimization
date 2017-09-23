#!/usr/bin/env Rscript

options(warn = -1)
suppressMessages(library(ggplot2))
suppressMessages(library(magrittr))
suppressMessages(library(dplyr))
suppressMessages(library(reshape2))
options(warn = 0)

# plot the convergence of Bayesian Optimization algorithms
setwd('~/Dropbox/code_base/BayesOpt')

args <- commandArgs(trailingOnly = TRUE)
csv <- paste0('ls ', args[1], '/*csv')

# plot a single algorithm
if (11 < 2) {
  colors <- c('#FF4F33', '#335EFF', '#FFE633', '#33FFDD')
  i <- 1
  
  for (file in system(csv, intern = TRUE)) {
    
    df <- read.csv(file) 
    N <- ncol(df)
    df.melt <- df %>% mutate(step = seq(nrow(.))) %>% melt(id = 'step')
    
    c <- colors[i]
    p <- ggplot(df.melt, aes(step, value)) + 
      stat_summary(geom = 'point', fun.y = mean, colour = c) + 
      stat_summary(geom = 'line', fun.y = mean, colour = c) + 
      stat_summary(geom = 'ribbon', 
                   fun.ymin = function(x) mean(x) - sd(x) / sqrt(N),
                   fun.ymax = function(x) mean(x) + sd(x) / sqrt(N), 
                   fill = c, alpha = 0.15)
    i <- i + 1
  }
  
  p <- p + scale_y_log10()
  print(p)
}

data <- NULL

for (file in system(csv, intern = TRUE)) {

  name <- strsplit(basename(file), '\\.')[[1]][1]
  cat(name, '\n')
  
  df <- read.csv(file) %>%
    tbl_df %>%
    mutate(algorithm = name) %>%
    mutate(step = seq(nrow(.))) %>% 
    melt(id = c('step', 'algorithm'), value.name = 'objective')

  if (is.null(data)) {
    data <- df
  } else {
    data <- rbind(data, df)
  }
}

p <- ggplot(data, aes(x = step, y = objective, colour = algorithm)) +
  stat_summary(geom = 'line', fun.y = mean) +
  stat_summary(geom = 'point', fun.y = mean) +
  stat_summary(geom = 'ribbon', aes(fill = algorithm), colour = NA,
               fun.ymin = function(x) mean(x) - sd(x) / sqrt(length(x)),
               fun.ymax = function(x) mean(x) + sd(x) / sqrt(length(x)), 
               alpha = 0.2)
p <- p + theme(legend.position = "bottom",
               legend.title = element_blank()) + 
    scale_y_log10() +
    labs(y = 'objective value')

ggsave('./plot.pdf', p, device = cairo_pdf(), height = 8, width = 12, dpi = 500)
