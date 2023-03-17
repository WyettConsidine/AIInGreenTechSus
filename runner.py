import getData

query = 'order: -announced_date_first; size: 200; hide_abstracts: True; include_cross_list: True; terms: AND title=sustainable tech; AND title=sustainability'

df = getData.getData(query)
print(df)

