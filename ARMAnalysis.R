library(arules)
library(tidyverse)
library(arules)
library(stringr)
install.packages('arulesViz')
library(arulesViz)

data <- read.csv("C:/Users/wyett/OneDrive/Documents/CSCI5502/AIInGreenTechSus/cleanedCompedData.csv")

head(data)

############################
#Using All Files:
transtotal <- arules::read.transactions("CSCI5502/AIInGreenTechSus/cleanedCompedData.csv",
                  rm.duplicates = TRUE, 
                  format = "basket",  ##if you use "single" also use cols=c(1,2)
                  sep=" ",  ## csv file
                  cols=NULL) ## The dataset has no row numbers

print(typeof(transtotal))

rules = arules::apriori(transtotal, parameter = list(support = 0.2,confidence = 0.6), maxlen=5)
print(length(rules))
rules <- subset(rules, subset = arules::size(arules::lhs(rules)) != 0)
print(arules::inspect(rules))

plot(rules, method="graph")


###############################

#Do ARM + Vis for Green Tech articles related to AI:
transGr <- arules::read.transactions("CSCI5502/AIInGreenTechSus/GreenTxtClean.csv",
                                   rm.duplicates = TRUE, 
                                   format = "basket",  ##if you use "single" also use cols=c(1,2)
                                   sep=" ",  ## csv file
                                   cols=NULL) ## The dataset has no row numbers

rulesG = arules::apriori(transGr, parameter = list(support = 0.8,confidence = 0.4), maxlen=5)
print(length(rulesG))
rulesG <- subset(rulesG, subset = arules::size(arules::lhs(rulesG)) != 0)
arules::inspect(rulesG)
plot(rulesG, method = "graph")

#######################################
#Do ARM + Vis for Sus articles related to AI:
transS <- arules::read.transactions("CSCI5502/AIInGreenTechSus/SusTxtClean.csv",
                                     rm.duplicates = TRUE, 
                                     format = "basket",  ##if you use "single" also use cols=c(1,2)
                                     sep=" ",  ## csv file
                                     cols=NULL) ## The dataset has no row numbers

rulesS = arules::apriori(transS, parameter = list(support = 0.40,confidence = 0.2), maxlen=5)
print(length(rulesS))
rulesS <- subset(rulesS, subset = arules::size(arules::lhs(rulesS)) != 0)
arules::inspect(rulesS)

plot(rulesS, method = "graph")

#####################################
#Do ARM + Vis for AI articles related to sus and Green T:
transA <- arules::read.transactions("CSCI5502/AIInGreenTechSus/AITxtClean.csv",
                                    rm.duplicates = TRUE, 
                                    format = "basket",  ##if you use "single" also use cols=c(1,2)
                                    sep=" ",  ## csv file
                                    cols=NULL) ## The dataset has no row numbers

rulesA = arules::apriori(transA, parameter = list(support = 0.15,confidence = 0.84), maxlen=5)
print(length(rulesA))
rulesA <- subset(rulesA, subset = arules::size(arules::lhs(rulesA)) != 0)
arules::inspect(rulesA)

plot(rulesA, method = "graph")

  