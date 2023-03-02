# This script is Wyett's attempt at reading in data from the API focussed on Sustainable Technology.
#list of websites that have good sustainability articles:
#https://www.mckinsey.com/capabilities/sustainability/our-insights/sustainability-blog/these-9-technological-innovations-will-shape-the-sustainability-agenda-in-2019


import pandas as pd
import numpy as np
import urllib, urllib.request

def getData():
    print("Getting Data")
    #all:electron&start=0&max_results=1
    url = 'http://export.arxiv.org/api/query?search_query=include_cross_list:True&terms:&all=sustainable&all=sustainability&all=environment'
    data = urllib.request.urlopen(url)
    print(type(data))
    print(data.read())
getData()
