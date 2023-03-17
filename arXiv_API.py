# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 17:15:09 2023

@author: gadge
"""

import pandas as pd
import numpy as np
import seaborn as sns
import requests
import time
import pandas_read_xml as pdx
import xml.etree.ElementTree as ET
import re

## Data Gathering

path = "C:/Users/gadge/OneDrive/Desktop/Data Mining/Project"

endpoint = "http://export.arxiv.org/api/query?"

topic = "ai"

url = endpoint+"search_query=all:"+topic+"&sortBy=submittedDate&sortOrder=descending&max_results=10000"

response = requests.get(url)

root = ET.fromstring(response.content)

entries = []
dictionary = {}

for child in root.iter('*'):
    tag = re.sub("\{.*?\}","",child.tag)
    if tag == 'entry':
        entries.append(dictionary)
        dictionary = {}
    if tag in ['id','updated','published','title','summary','primary_category']:
        dictionary[tag] = child.text
        
df = pd.DataFrame(entries)

df['summary'] = df['summary'].apply(lambda x: str(x).replace('\n',' '))
df['summary'] = df['summary'].apply(lambda x: str(x).replace('\t',' '))

df.to_csv(path+'/arXiv_AI.txt',sep = '\t',index = False)

df = pd.read_csv(path+'/arXiv_AI.txt',sep = '\t')

## Data cleaning

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import re
lemmatizer = WordNetLemmatizer()

def text_clean(data):
    data = re.sub( '[^A-Za-z]', ' ', data)
    data = re.sub(' +', ' ', data).lower()
    word_list = word_tokenize(data)
    data = ' '.join([lemmatizer.lemmatize(w) for w in word_list if len(w) > 3])
    return data
    
df['summary_clean'] = df['summary'].apply(text_clean)

## word cloud

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

text = df['summary_clean'].apply(lambda x: ' '.join([i for i in x.split(" ") if (len(i) > 4)]))
text = ' '.join([i for i in text.tolist() if len(i) > 3])

wordcloud = WordCloud(stopwords = STOPWORDS,
                      collocations=True,
                      width = 1000, height = 1000,
                      background_color ='white',
                      max_words = 120).generate(text)


plt.imshow(wordcloud, interpolation='bilInear')
plt.axis('off')


