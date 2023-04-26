# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 19:05:20 2023

@author: gadge
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import re
from nltk.stem import PorterStemmer
import string


df = pd.read_csv("https://raw.githubusercontent.com/WyettConsidine/AIInGreenTechSus/master/complete_data.csv")
df.memory_usage()
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


greentech_sus = pd.concat([green_tech,Sus])

greentech_sus['Month'] = greentech_sus['Date'].dt.to_period('M')
ai['Month'] = ai['Date'].dt.to_period('M')

greentech_sus_month = greentech_sus.groupby('Month')[['AI']].sum().reset_index()
ai_month = ai.groupby('Month')[['GreenTech Sustainability']].sum().reset_index()

greentech_sus_month = greentech_sus_month.set_index('Month')
ai_month = ai_month.set_index("Month")

path = "C:/Users/chaub/Documents/CU_Boulder/Spring 2023/CSCI 5502 Data Mining/Data Mining Project/AIInGreenTechSus"

greentech_sus_month.plot()
plt.xlabel("Month")
plt.ylabel("Number of research papers and articles")
plt.title(" AI based research articles in Green Tech and Sustainability ")
plt.savefig(path +"/AIinGreentechandsus.png")

ai_month.plot()
plt.xlabel("Month")
plt.ylabel("Number of research papers and articles")
plt.title(" Green Tech and Sustainability based articles in AI ")
plt.savefig(path +"/GreentechandsusinAi.png")



#***********************************************************************#

green_tech.head()
green_tech.info()
print(f'Shape of green tech data : {green_tech.shape}')
gt = green_tech[green_tech['AI']]
gt = gt[['Title', 'Date', 'Content', 'Topic']]
print(f'Shape of subsetted green tech data : {gt.shape}')

Sus.head()
print(f'Shape of sustainable data : {Sus.shape}')
s = Sus[Sus['AI']]
s.columns
s = s[['Title', 'Date', 'Content', 'Topic']]
print(f'Shape of subsetted sustainability data : {s.shape}')

ai.head()
ai.columns
print(ai.shape)
ai = ai[ai['GreenTech Sustainability']]
ai = ai[['Title', 'Date', 'Content', 'Topic']]
print(ai.shape)

#*******TEXT SUMMARIZATION USING gensim*******************************#

#Creating a function to convert article content into summary 
#for green tech articles

def clean_string(line):
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

gt['clean'] = gt['Content'].apply(clean_string)
s['clean'] = s['Content'].apply(clean_string)
ai['clean'] = ai['Content'].apply(clean_string)

    

def summarize(text, preprocessed_text):
    
    o_sentences = sent_tokenize(text)

    # Tokenize the text into sentences
    sentences = sent_tokenize(preprocessed_text)
    
    # Tokenize each sentence into words and filter out stop words
    stop_words = set(stopwords.words("english"))
    words = [word_tokenize(sentence.lower()) for sentence in sentences]
    filtered_words = [[word for word in sentence if word not in stop_words] for sentence in words]
    
    # Stem the words using Porter Stemmer
    stemmer = PorterStemmer()
    stemmed_words = [[stemmer.stem(word) for word in sentence] for sentence in filtered_words]
    
    # Calculate the frequency of each word
    word_frequencies = {}
    for sentence in stemmed_words:
        for word in sentence:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    
    # Calculate the weighted frequency of each sentence
    sentence_scores = {}
    for i, sentence in enumerate(stemmed_words):
        for word in sentence:
            if word in word_frequencies:
                if i not in sentence_scores:
                    sentence_scores[i] = word_frequencies[word]
                else:
                    sentence_scores[i] += word_frequencies[word]
    
    # Get the top N sentences with the highest weighted frequency
    N = 2
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:N]
    
    # Create the summary
    summary = " ".join([o_sentences[i] for i in top_sentences])
    
    # Print the summary
    return summary


gt['Summary'] = gt.apply(lambda x: summarize(x.Content, x.clean), axis = 1)
s.rename({'Content': 'Summary'}, axis = 1, inplace = True)
ai.rename({'Content': 'Summary'}, axis = 1, inplace = True)




