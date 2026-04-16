## Script - Maria Laura (Moderacao e analise de caminhos)
library(lavaan)
library(knitr)
library(tidyverse)

# Set working directory
setwd("H:/Meu Drive/parteI/machine_learning/v15")

# Load data frame
data1 <- read.csv("sensitivity.tsv", header=TRUE, sep='\t')

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


parameterEstimates(model_sem, ci = TRUE, level = 0.95, boot.ci.type = "perc")

boot_est <- lavInspect(model_sem, "boot")
hist(boot_est[, "b3"], breaks=50, xlim=c(-0.15,0.5), ylim=c(0,60))
