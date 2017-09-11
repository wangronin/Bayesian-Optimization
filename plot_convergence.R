library(ggplot2)
library(magrittr)
library(dplyr)
library(reshape2)

setwd('~/Dropbox/code_base/Configurator/')

if (1 < 2) {
  colors <- c('#FF4F33', '#335EFF', '#FFE633', '#33FFDD')
  i <- 1
  
  for (file in system('ls ./data/*csv', intern = TRUE)) {
    
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

# data <- NULL
# for (file in system('ls *csv', intern = TRUE)) {
#   
#   name <- strsplit(file, '\\.')[[1]][1]
#   df <- read.csv(file) %>%
#     tbl_df %>%
#     mutate(algorithm = name) %>%
#     melt(id = c('step', 'algorithm'), value.name = 'convergence')
#   
#   if (is.null(data)) {
#     data <- df
#   } else {
#     data <- rbind(data, df)
#   }
# }
# 
# p <- ggplot(data, aes(x = step, y = convergence, colour = algorithm)) + 
#   stat_summary(geom = 'line', fun.y = mean) +
#   stat_summary(geom = 'ribbon', aes(fill = algorithm), colour = NA, 
#                fun.ymin = function(x) mean(x)-sd(x)/4,
#                fun.ymax = function(x) mean(x)+sd(x)/4, alpha=0.15)
# p <- p + scale_y_log10() + theme(legend.position = "bottom")
# 
# # ggsave('test.pdf', p, device = cairo_pdf(), height = 8, width = 12)
# print(p)
