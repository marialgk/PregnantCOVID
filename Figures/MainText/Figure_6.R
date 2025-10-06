## Moderation analysis 

library(dplyr)
library(knitr)
library(ggplot2)


# Set working directory
setwd("H:/Meu Drive/parteI/machine_learning/v14/SEM")

# Load data frame
data <- read.csv("data.tsv", header=TRUE, sep='\t')

# 1. Ajustar modelo com interação
model <- glm(delivery_complications ~ STAIT * group, family = binomial, data = data)

# 2. Obter probabilimomes preditas
data$predicted_prob <- predict(model, type = "response")

# 3. Criar tercis de STAI_trait
data$STAI_tercil <- cut(data$mom_age,
                        breaks = quantile(data$STAIT, probs = c(0, 1/3, 2/3, 1)),
                        include.lowest = TRUE,
                       labels = c("Lower", "Moderate", "Higher"))

# 4. Converter COVID_status para rótulo legível
data$group_str<- factor(data$group, labels = c("CONTROL", "COVID"))

# 5. Fazer boxplot
ggplot(data, aes(x = STAI_tercil, y = predicted_prob, fill = group_str)) +
  geom_boxplot() +
  labs(
    y = "Predicted probability of complications",
    x = "Trait anxiety",
    fill = "COVID"
  ) +
  theme_minimal()
