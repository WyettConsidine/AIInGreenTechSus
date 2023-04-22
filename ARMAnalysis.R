library(arules)
library(tidyverse)
library(arules)
library(stringr)

data <- read.csv("CSCI5502/AIInGreenTechSus/cleanedCompedData.csv")



#Using All Files:
transtotal <- arules::read.transactions("CSCI5502/AIInGreenTechSus/cleanedCompedData.csv",
                  rm.duplicates = TRUE, 
                  format = "basket",  ##if you use "single" also use cols=c(1,2)
                  sep=",",  ## csv file
                  cols=NULL) ## The dataset has no row numbers

rules = arules::apriori(transtotal, parameter = list(support = 0.03,confidence = 0.60), maxlen=5)
rules <- subset(rules, subset = arules::size(arules::lhs(rules)) != 0)
arules::inspect(rules)

#Get All Most Frequent Words in Green T : Provides No Useful Rules: Create custom List
transGr <- arules::read.transactions("CSCI5502/AIInGreenTechSus/GreenTxtClean.csv",
                                   rm.duplicates = TRUE, 
                                   format = "basket",  ##if you use "single" also use cols=c(1,2)
                                   sep=",",  ## csv file
                                   cols=NULL) ## The dataset has no row numbers

rulesG = arules::apriori(transGr, parameter = list(support = 0.01,confidence = 0.001), maxlen=5)
print(length(rulesG))
rulesG <- subset(rulesG, subset = arules::size(arules::lhs(rulesG)) == 0)
arules::inspect(rulesG)

#Get All Most Frequent Words in Sus Tech Articles :
transSus <- arules::read.transactions("CSCI5502/AIInGreenTechSus/SusTxtClean.csv",
                                     rm.duplicates = TRUE, 
                                     format = "basket",  ##if you use "single" also use cols=c(1,2)
                                     sep=",",  ## csv file
                                     cols=NULL) ## The dataset has no row numbers

rulesSus = arules::apriori(transSus, parameter = list(support = 0.008,confidence = 0.001), maxlen=5)
print(length(rulesSus))
#rulesG <- subset(rulesSus, subset = arules::size(arules::lhs(rulesSus)) == 0)
arules::inspect(rulesSus)



  