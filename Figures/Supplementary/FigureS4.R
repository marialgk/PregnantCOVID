## Script - Maria Laura (Moderacao e analise de caminhos)
library(rockchalk)
library(boot)
library("psych")
library(lavaan)
library(knitr)
library(semPlot)
library(tidyverse)
library(rquery)
library(tidySEM)
library(simpleboot)

data <- read.csv("data.tsv", header=TRUE, sep='\t')

# Gráfico
# Calcular a média e desvio padrão de STAIT
mean_stait <- mean(data$STAIT, na.rm = TRUE)
sd_stait <- sd(data$STAIT, na.rm = TRUE)

# Criar quatro níveis de STAIT: (m-sd), m, (m+sd), acima de (m+sd) e abaixo de (m-sd)
data$COVID <- factor(data$group)

# Plotar o gráfico com ggplot para visualizar a moderação de STAIT
ggplot(data) +
  geom_point(aes(x = mom_age, y = delivery_complications, color = COVID), alpha = 0.6) +  # Pontos
  geom_smooth(aes(x = mom_age, y = delivery_complications, color = COVID), method = "glm", method.args = list(family = "binomial"), se = FALSE) +  # Linha de regressão
  labs(x = "Maternal Age",
       y = "Delivery complications",
       color = "COVID-19") +
  theme(legend.position = "top")