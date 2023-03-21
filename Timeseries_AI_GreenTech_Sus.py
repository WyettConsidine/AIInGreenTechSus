# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 19:05:20 2023

@author: gadge
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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


greentech_sus = pd.concat([green_tech,Sus])

greentech_sus['Month'] = greentech_sus['Date'].dt.to_period('M')
ai['Month'] = ai['Date'].dt.to_period('M')

greentech_sus_month = greentech_sus.groupby('Month')[['AI']].sum().reset_index()
ai_month = ai.groupby('Month')[['GreenTech Sustainability']].sum().reset_index()

greentech_sus_month = greentech_sus_month.set_index('Month')
ai_month = ai_month.set_index("Month")

path = "C:/Users/gadge/OneDrive/Desktop/Data Mining/Project"

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