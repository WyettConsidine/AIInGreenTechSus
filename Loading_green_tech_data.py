# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:09:17 2023

@author: chaub
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt



# List for the ending portion of the link to go to new pages
 

page_num = ['','P25', 'P50', 'P75', 'P100', 'P125', 'P150', 'P175','P200', 'P225', 'P250', 'P275', 'P300', 'P325', 'P350', 'P375', 'P400', 'P425', 'P450', 'P475', 'P500', 'P525', 'P550', 'P575', '600']
url_start = 'https://www.greentechmedia.com/'

#Creating a list to store the article links from all pages
article_link_list  =[]

#Creating lists to store the article title, date, author and content
article_title_list = []
article_author_list = []
article_date_list = []
article_content_list = []

#Creating a list to remove links that contain the below words, because they are
#not links to the article but to a new page with more articles.
matches = ["squared", "sponsored", "research-spotlight", "industry-perspective"]

#Looping through the pages to extract article links and storing them in article_link_list
for i in range(len(page_num)):
    complete_url = url_start + page_num[i]
    # print(complete_url)

    response = requests.get(complete_url)
    #uncomment below for progress bar
    
    print(response, 'here')

    page_soup = BeautifulSoup(response.content, 'html.parser')

    for item in page_soup.find_all('div', class_='article-detail'):
        #print('here_now')
        article_link = item.find('a')
        specific_article_link = article_link.get('href')
        article_link_start = 'https://www.greentechmedia.com'
        complete_article_link = article_link_start+specific_article_link
        #print(complete_article_link)
        #skipping article links that contains any string in matches list above
        if any([x in complete_article_link for x in matches]):
            continue 
        
        article_link_list.append(complete_article_link)
#checking length of list i.e number of atricle links
print(len(article_link_list))
print(article_link_list)


#Iterating through each atricle link in article_link_list to extract title, date, author, content

for link in article_link_list:
    
           
    #print('Getting info about', link)
    article_response = requests.get(link)
    #print(article_response)
    
    
    article_soup = BeautifulSoup(article_response.content, 'html.parser')
    #print(article_soup)
    #print(type(article_soup))
    
    author_soup=article_soup.find_all('span', class_ = 'article-author')    
    if author_soup == []:
        continue
    for author_n in author_soup:
        author_name=author_n.get_text()
        article_author_list.append(author_name)
       
    
    
    try:
        article_title = article_soup.find('h1').get_text()
        #print(article_title)             
        article_title_list.append(article_title)
    except AttributeError:
        pass
    
    
    date_soup=article_soup.find_all('span', class_ = 'article-date')
    for date_n in date_soup:
        try:
            date=date_n.get_text()
            article_date_list.append(date)
        except AttributeError:
            pass
        

    article_body_soup=article_soup.find_all('div', class_ = 'col-md-9 article-content first-article-content')
    for body in article_body_soup:
        para = body.find_all('p')
        new_para = []
        for paragraph in para:
            article_content = paragraph.get_text()
            new_para.append(article_content)
        
        article_content_list.append(new_para)
    print(article_link_list.index(link))
    if len(article_title_list) == len(article_author_list):
        print('True')
    else:
        print('False')
        
print(len(article_link_list))
#Checking lengths of all lists to confirm the lists are of the same length

print(len(article_title_list))
print(len(article_author_list))
print(len(article_date_list))
print(len(article_content_list))

print(article_title_list)


green_tech_data = pd.DataFrame()
green_tech_data['Title'] = article_title_list 
green_tech_data['Author'] = article_author_list
green_tech_data['Date'] = article_date_list
green_tech_data['Content'] = article_content_list

green_tech_data.head()

#Convert content to a string from list of string
def make_one_string(list):
    return(' '.join(list))

green_tech_data['Content'] = green_tech_data['Content'].apply(make_one_string)
 
#removing all punctuations and nums from title and content 
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
    
    return(line)

green_tech_data['Title'] = green_tech_data['Title'].apply(clean_string)
green_tech_data['Content'] = green_tech_data['Content'].apply(clean_string)

#Removing stopwords from title and content
def remove_stopwords(line):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(line)
    line_filtered = [w for w in word_tokens if not w.lower() in stop_words]
    
    return ' '.join(line_filtered)
    
#test_line = 'at the beginning of the pandemic energy prices crashed we did anepisode of this showtrying to figure out how oil prices fell to negative per barreltimes have changed oil is up over barrel but far more acute is '#what’s happening with natural gas particularly in europe and asia in the us natural gas prices have doubled in the last year but in parts of europe the price has risen more than timesthe disruptions are clear were seeing stories of power shortages in china fertilizer plants being shut down in the uk and fears about home heating costs in the northeast us as winter approachesso what the heck is going on how long might it last and what does it tell us about the futureto answer those questions shayle turns to leslie paltiguzmanthe president of gas vista and a nonresident fellowat nyu sps center for global affairsshayle and leslie cover the many demandside and supplyside issues then they talk about what comes next what does this crisis reveal about the vulnerability of the energy system and will countries double down on renewables gas or both to shore up their resiliencythe interchangeis brought to you by schneider electric are you building a microgrid with a microgrid you can store electricity and sell it back during peak times keep your power on during an outage integrate with renewables control energy on your own terms having built more microgrids in than anyone elseschneider electrichas the expertise to helpthe interchangeis brought to you by bloom energy bloom’s onsite energy platform provides unparalleled control for those looking to secure clean reliable power that scales to meet critical business needs it eliminates outage and price risk while accelerating us towards a zero carbon future visitbloom energyto learn how to take charge today times have changed oil is up over barrel but far more acute is what’s happening with natural gas particularly in europe and asia in the us natural gas prices have doubled in the last year but in parts of europe the price has risen more than times the disruptions are clear were seeing stories of power shortages in china fertilizer plants being shut down in the uk and fears about home heating costs in the northeast us as winter approaches so what the heck is going on how long might it last and what does it tell us about the future to answer those questions shayle turns to leslie paltiguzmanthe president of gas vista and a nonresident fellowat nyu sps center for global affairs shayle and leslie cover the many demandside and supplyside issues then they talk about what comes next what does this crisis reveal about the vulnerability of the energy system and will countries double down on renewables gas or both to shore up their resiliency the interchangeis brought to you by schneider electric are you building a microgrid with a microgrid you can store electricity and sell it back during peak times keep your power on during an outage integrate with renewables control energy on your own terms having built more microgrids in than anyone elseschneider electrichas the expertise to help the interchangeis brought to you by bloom energy bloom’s onsite energy platform provides unparalleled control for those looking to secure clean reliable power that scales to meet critical business needs it eliminates outage and price risk while accelerating us towards a zero carbon future visitbloom energyto learn how to take charge today'
#print(remove_stopwords(test_line))

green_tech_data['Content'] = green_tech_data['Content'].apply(remove_stopwords)
green_content = ' '.join(green_tech_data['Content'])
word_cloud = WordCloud(collocations = False, background_color = 'white').generate(green_content)
plt.imshow(word_cloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()

#Adding label column with value 'green_tech'
green_tech_data['Label'] = 'green_tech'

#checking columns
green_tech_data.head()

#saving data as text
green_tech_data.to_csv('C:/Users/chaub/Documents/CU_Boulder/Spring 2023/CSCI 5502 Data Mining/Data Mining Project/AIInGreenTechSus/green_tech.txt', sep = '\t', index = False)


















