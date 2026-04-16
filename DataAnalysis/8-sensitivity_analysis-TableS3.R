## Moderation analysis 

library(dplyr)
library(lavaan)
library(knitr)

# Set working directory
setwd("H:/Meu Drive/parteI/machine_learning/v15")

# Load data frame
data1 <- read.csv("sensitivity.tsv", header=TRUE, sep='\t')

data1 <- data1 %>% mutate(ModT = group * STAIT)

# mom_age, prenatal_bmi, works, mom_education, race_marginalized_group, income

mod_model = '
delivery_complications ~ b1*group + b2*STAIT + b3*ModT + b4*works

STAIT ~ STAIT.mean*1
STAIT ~~ STAIT.var*STAIT

# Derived parameters
SD.below := b1 + b3 * (STAIT.mean - sqrt(STAIT.var))
mean := b1 + b3 * STAIT.mean
SD.above := b1 + b3 * (STAIT.mean + sqrt(STAIT.var))
'

model_sem = sem(mod_model, data=data1, 
                estimator="DWLS", 
                se='boot', bootstrap=1000)

summary(model_sem, standardized=TRUE, rsq=T) 

parameterEstimates(model_sem, ci = TRUE, level = 0.95, boot.ci.type = "perc")
