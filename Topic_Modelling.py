# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 12:37:22 2023

@author: gadge
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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


df = pd.read_csv("https://raw.githubusercontent.com/WyettConsidine/AIInGreenTechSus/master/complete_data.csv")

df['Date'] = pd.to_datetime(df['Date'])
df['Content'] = df['Content'].astype('str')

green_tech = df[df['Topic'] == "Green_Tech"]
Sus = df[df['Topic'] == 'Sustainability']
ai = df[df['Topic'] == "Artificial_Intellgence"]

ai_terms = ['artificial intelligence','machine learning']
ai_words = ['ai','ml']


green_tech['ai_1'] = green_tech['Content'].apply(lambda x:len([i for i in ai_terms if i in x]))

green_tech['ai_2'] = green_tech['Content'].apply(lambda x: len([i for i in x.split(" ") if i in ai_words]))

green_tech['ai'] = green_tech['ai_1']+green_tech['ai_2']

green_tech['AI'] = green_tech['ai'] > 0

Sus['ai_1'] = Sus['Content'].apply(lambda x:len([i for i in ai_terms if i in x]))

Sus['ai_2'] = Sus['Content'].apply(lambda x: len([i for i in x.split(" ") if i in ai_words]))

Sus['ai'] = Sus['ai_1']+Sus['ai_2']

Sus['AI'] = Sus['ai'] > 0

greentech_sus_words = ['green tech','green technology','environment','sustainable','sustainability']

ai['GreenTech Sustainability'] = ai['Content'].apply(lambda x: len([i for i in greentech_sus_words if i in x])> 0)

green_tech['Year'] = green_tech['Date'].dt.to_period('Y')
ai['Year'] = ai['Date'].dt.to_period('Y')
Sus['Year'] = Sus['Date'].dt.to_period('Y')

lemmatizer = WordNetLemmatizer()
def text_clean(data):
    data = re.sub('[^A-Za-z]', ' ', data) #remove special char
    data = re.sub(' +', ' ', data).lower() #remove mulitple spaces
    word_list = word_tokenize(data) #tokenize
    data = ' '.join([lemmatizer.lemmatize(w) for w in word_list if w != 'nan']) #if len(w) > 3 # lemmanize words
    return data

ai['summary_clean'] = ai['Content'].apply(text_clean)
green_tech['summary_clean'] = green_tech['Content'].apply(text_clean)
Sus['summary_clean'] = Sus['Content'].apply(text_clean)

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD



path = "C:/Users/gadge/OneDrive/Desktop/Data Mining/Project"
def topic_modelling(docs, num_topics, top_n, filename):
    vectorizer = TfidfVectorizer(stop_words = 'english',max_df=0.9)
    X = vectorizer.fit_transform(docs).toarray()
    words = vectorizer.get_feature_names_out()
    TFDF = pd.DataFrame(X, columns=words)
    lda_model_DH = LatentDirichletAllocation(n_components=num_topics, 
                                         max_iter=100, learning_method='online')
    LDA_DH_Model = lda_model_DH.fit_transform(TFDF)
    j = 0
    
    for idx, topic in enumerate(lda_model_DH.components_):
        print("Topic:  ", idx)
      
        print([(words[i], topic[i])for i in topic.argsort()[:-top_n - 1:-1]])
        top_words = [words[i] for i in topic.argsort()[:-top_n - 1:-1]]
        top_words_shares = [topic[i] for i in topic.argsort()[:-top_n - 1:-1]]
        plt.subplot(1, num_topics, j + 1)  # plot numbering starts with 1
        plt.ylim(0, top_n + 0.5)  # stretch the y-axis to accommodate the words
        plt.xticks([])  # remove x-axis markings ('ticks')
        plt.yticks([]) # remove y-axis markings ('ticks')
        plt.title('Topic #{}'.format(j))
        
        for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
            plt.text(0.3, top_n-i-0.5, word, fontsize=8)
        j = j+1
    
    #plt.title(filename)      
    plt.tight_layout()
    
    plt.savefig(path +"/"+filename+".pdf",format = 'pdf')
    plt.savefig(path +"/"+filename+".png",format = 'png')
    plt.show()
    


topic_modelling(ai['summary_clean'], 6, 10,'AI')
topic_modelling(green_tech['summary_clean'], 6, 10,'Green_tech')
topic_modelling(Sus['summary_clean'], 6, 10,'Sustainability')  
  
for year in ai['Year'].unique():
    ai_year = ai[ai['Year'] == year]
    filename = 'AI '+str(year)
    topic_modelling(ai_year['summary_clean'], 6, 10,filename)
    
for year in green_tech['Year'].unique():
    gt_year = green_tech[green_tech['Year'] == year]
    filename = 'GreenTech '+str(year)
    topic_modelling(gt_year['summary_clean'], 6, 10,filename)

for year in Sus['Year'].unique():
    sus_year = Sus[Sus['Year'] == year]
    filename = 'Sus '+str(year)
    topic_modelling(sus_year['summary_clean'], 6, 10,filename)
    
ai_gtsus = ai[ai['GreenTech Sustainability']]
topic_modelling(ai_gtsus['summary_clean'], 6, 10,"AI_with_greentechsus")

gt_ai = green_tech[green_tech['AI']]
topic_modelling(gt_ai['summary_clean'], 6, 10,"greentech_with_ai")

sus_ai = Sus[Sus['AI']]
topic_modelling(sus_ai['summary_clean'], 6, 10,"sus_with_ai")

