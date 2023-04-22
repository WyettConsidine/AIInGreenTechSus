# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 11:35:05 2023

@author: wyett
"""

import numpy as np
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
#nltk.download('stopwords')


df = pd.read_csv(r"C:\Users\wyett\OneDrive\Documents\CSCI5502\AIInGreenTechSus\complete_data.csv", 
                 index_col = 'id',
                 usecols = range(5))


def clean_string(line):
    #print(line)
    line = str(line)
    line = line.lower()
    line = re.sub(r'\d+', '', line)
    line = re.sub('—',' ', str(line))
    table = line.maketrans('','',string.punctuation)
    line = line.translate(table)
    line = re.sub('\n|\r', '', line)
    line = re.sub(r' +', ' ', line)
    line = re.sub('\xa0', '', line)
    line = re.sub('\n', '', line)
    line = re.sub('’', '', line)
    
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(line)
    line_filtered = [w for w in word_tokens if not w.lower() in stop_words]
    
    return ' '.join(line_filtered)

df['CleanContent'] = [clean_string(line) for line in df['Content']]

df.to_csv(r"C:\Users\wyett\OneDrive\Documents\CSCI5502\AIInGreenTechSus\cleanedCompedData.csv")

AITxt = df[ df['Topic'] == 'Artificial_Intellgence']
GreenTxt = df[ df['Topic'] == 'Green_Tech']
SusTxt = df[ df['Topic'] == 'Sustainability']

print(AITxt.columns)


#df.to_csv(r"C:\Users\wyett\OneDrive\Documents\CSCI5502\AIInGreenTechSus\transactionalData.csv")
