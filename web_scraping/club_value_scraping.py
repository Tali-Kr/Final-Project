from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm

links = []
temp = []
clubs = []
values = []
squad = []

for year in list(range(2012, 2022)):
    for month in ('{:02d}'.format(i) for i in range(1, 13)):
        for day in ['01', '15']:
            links.append('https://www.transfermarkt.com/ligat-haal/marktwerteverein/wettbewerb/ISR1/stichtag/' +
                         str(year) + '-' + month + '-' + day+'/plus/1')

df = pd.DataFrame(columns=['date', 'club', 'value', 'squad_s'])

headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/'
                         '537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
errors = 0
for link in tqdm(links):
    attempt = 0
    while attempt != 6:
        try:
            links = []
            temp = []
            clubs = []
            values = []
            squad = []

            page = link
            pageTree = requests.get(page, headers=headers)
            pageSoup = BeautifulSoup(pageTree.content, 'html.parser')

            # Clubs
            for g1 in pageSoup.find_all('td', class_='hauptlink no-border-links'):
                clubs.append(g1.text)

            # Values
            for g1 in pageSoup.find_all('td', class_='rechts'):
                for g2 in g1.find_all('a'):
                    temp.append(g2.text.replace("€", "").replace(".", "").replace("m", "0000").replace("k", "000").
                                replace("-", "0"))

            for i, g1 in enumerate(temp):
                if i % 3 == 0:
                    values.append(g1)
            temp = []

            # Date
            date = (link.split("/")[8].split("-")[2]+'/'+link.split("/")[8].split("-")[1]+'/'+link.split("/")[8].
                            split("-")[0])

            # Squad size
            for g1 in pageSoup.find_all('td', class_='zentriert'):
                for g2 in g1.find_all('a'):
                    temp.append(g2.text)

            counter = 1
            while True:
                try:
                    if counter == 1:
                        squad.append(temp[counter])
                        counter += 1
                    counter += 5
                    squad.append(temp[counter])
                except IndexError:
                    break
            for i in range(0, 14):
                df.loc[len(df)] = [date, clubs[i], values[i], squad[i]]
            attempt = 6
        except Exception:
            errors += 1
            if attempt == 5:
                attempt += 1
print('Num of errors occurred:', errors)
df.to_csv('test.csv')