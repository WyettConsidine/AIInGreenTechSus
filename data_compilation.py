import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import string
import re



ai_data = pd.read_csv("C:/Users/chaub/Documents/CU_Boulder/Spring 2023/CSCI 5502 Data Mining/Data Mining Project/AIInGreenTechSus/arXiv_AI.txt", sep = '\t')
g_data = pd.read_csv("C:/Users/chaub/Documents/CU_Boulder/Spring 2023/CSCI 5502 Data Mining/Data Mining Project/AIInGreenTechSus/green_tech.txt", sep = '\t')
s_data = pd.read_csv("C:/Users/chaub/Documents/CU_Boulder/Spring 2023/CSCI 5502 Data Mining/Data Mining Project/AIInGreenTechSus/sus_data.txt", sep = '\t')

ai_data.head()
g_data.head()

#processing the AI data, changing column name, adding topic column
ai_data.columns
ai_data = ai_data[['title', 'updated', 'summary']]
ai_data['Topic'] = 'Artificial_Intellgence'
ai_data.rename(columns = {'title': 'Title', 'updated': 'Date', 'summary': 'Content'}, inplace = True)
ai_data.columns
ai_data.head()


#creating function to separate date from time
def strip_date(string):
    new_s = string[:10]
    return new_s

#Creating function to change text to date
def change_to_Date(string):
    datetime_object = datetime.strptime(string, '%Y-%m-%d').date()
    return datetime_object
    
ai_data['Date'] = ai_data['Date'].apply(strip_date)
ai_data['Date'] = ai_data['Date'].apply(change_to_Date)

ai_data.info()


#formatting green tech data

def change_to_Date2(str_date):
    table = str_date.maketrans('','', string.punctuation)
    str_date = str_date.translate(table)
    str_date = re.sub(' ','', str_date)
    input_format = '%B%d%Y'
    output_format = '%Y-%m-%d'
    
    datetime_object = datetime.strptime(str_date, input_format).strftime(output_format)
    
    return datetime_object

g_data.columns
g_data = g_data[['Title', 'Date', 'Content']]
g_data['Topic'] = 'Green_Tech'
g_data['Date'] = g_data['Date'].apply(change_to_Date2)

ai_data.shape
g_data.shape

df_partial = pd.concat([ai_data, g_data])
df_partial.shape

# printed in default format

sus_data.columns
sus_data = sus_data[[]]


matches = ['Artificial Intelligence', 'artificial intelligence', ' ai ',' AI ', 'Machine Learning', ' ml ', ' ML ']

for i in 
if any([x in complete_article_link for x in matches]):
    continue 










