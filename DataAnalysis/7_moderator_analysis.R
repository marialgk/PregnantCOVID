## Script - Maria Laura (Moderacao e analise de caminhos)
library(rockchalk)
library(boot)
library("psych")
library(lavaan)
library(knitr)
library(semPlot)
library(tidyverse)
library(rquery)
library(simpleboot)

# Set working directory
setwd("H:/Meu Drive/parteI/machine_learning/v14/SEM/4vars")

# Load data frame
data <- read.csv("data.tsv", header=TRUE, sep='\t')

### Package 'Lavaan' ###
# Creating interaction term prior of sem running
data <- data %>% mutate(Mod = group * STAIT)

mod_model = '
delivery_complications ~ b1*group + b2*STAIT + b3*Mod

STAIT ~ STAIT.mean*1
STAIT ~~ STAIT.var*STAIT

# Derived parameters
SD.below := b1 + b3 * (STAIT.mean - sqrt(STAIT.var))
mean := b1 + b3 * STAIT.mean
SD.above := b1 + b3 * (STAIT.mean + sqrt(STAIT.var))
'

model_sem = sem(mod_model, data=data, 
                estimator="DWLS", 
                se='boot', bootstrap=1000)

summary(model_sem, standardized=TRUE) 

fitMeasures(model_sem, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr"))
