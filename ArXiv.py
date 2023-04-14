# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:08:12 2023

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
from os import getcwd

## Data cleaning
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import re

#This function takes in a query from the arXiv UI website.
#https://arxiv.org/search/advanced

query = 'order: -announced_date_first; size: 200; hide_abstracts: True; include_cross_list: True; terms: AND title=sustainable tech; AND title=sustainability'

def getData(query):
    queryURL = "http://export.arxiv.org/api/query?search_query=" + query
    response = requests.get(queryURL)
    root = ET.fromstring(response.content)
    entries = []
    dictionary = {}

    for child in root.iter('*'):
        tag = re.sub("\{.*?\}", "", child.tag)
        if tag == 'entry':
            entries.append(dictionary)
            dictionary = {}
        if tag in ['id', 'updated', 'published', 'title', 'summary', 'primary_category']:
            dictionary[tag] = child.text

    df = pd.DataFrame(entries)

    df['summary'] = df['summary'].apply(lambda x: str(x).replace('\n', ' '))
    df['summary'] = df['summary'].apply(lambda x: str(x).replace('\t', ' '))
    return df

#This takes the output df from getData(), and cleans it
#Adds column to df: summary_clean.
#cleaning steps:
def cleanData(df):
    lemmatizer = WordNetLemmatizer()

    def text_clean(data):
        data = re.sub('[^A-Za-z]', ' ', data) #remove special char
        data = re.sub(' +', ' ', data).lower() #remove mulitple spaces
        word_list = word_tokenize(data) #tokenize
        data = ' '.join([lemmatizer.lemmatize(w) for w in word_list ]) #if len(w) > 3 # lemmanize words
        return data

    df['summary_clean'] = df['summary'].apply(text_clean)

    ## word cloud generation

    text = df['summary_clean'].apply(lambda x: ' '.join([i for i in x.split(" ") if (len(i) > 4)]))
    text = ' '.join([i for i in text.tolist() if len(i) > 3])

    wordcloud = WordCloud(stopwords=STOPWORDS,
                          collocations=True,
                          width=1000, height=1000,
                          background_color='white',
                          max_words=120).generate(text)

    plt.imshow(wordcloud, interpolation='bilInear')
    plt.axis('off')

    return df

path = path = "C:/Users/gadge/OneDrive/Desktop/Data Mining/Project"

def saveAsTxt(df, filename):
    path = os.getcwd()
    df.to_csv(path + '/' + filename, sep='\t', index=False)
query = 'order: -announced_date_first; size: 200; hide_abstracts: True; include_cross_list: True; terms: AND title=sustainable tech; AND title=sustainability'

df = getData(query)
print(df)