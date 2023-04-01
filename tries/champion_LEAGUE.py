from bs4 import BeautifulSoup
import requests
import re

headers = {'User-Agent':
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}

# url = "https://www.transfermarkt.com/bundesliga/startseite/wettbewerb/L1/plus/?saison_id=2015"
# response = requests.get(url, headers=headers)
# results = BeautifulSoup(response.content, 'html.parser')

# trs = results.find_all("tr", class_="odd")[0].find_all("a")[-1].get('title') #הצלחה !!!
# trs = results.find_all("tr", class_="odd")
# germany = []
# for tr in range(len(trs)):
#     germany.append(trs[tr].find_all("a")[-1].get('title'))

url_ = "https://www.transfermarkt.com/uefa-champions-league/startseite/pokalwettbewerb/CL?saison_id=2016"
response = requests.get(url_, headers=headers)
results = BeautifulSoup(response.content, 'html.parser')
trs = results.find_all("td", class_="no-border-links hauptlink")[0]
href = trs.find_all("a")[-1].get('href')
title = trs.find_all("a")[-1].get('title')
transfermrkt = "https://www.transfermarkt.com"
url_in = transfermrkt+href
response_ = requests.get(url_in, headers=headers)
results_ = BeautifulSoup(response_.content, 'html.parser')
trs_ = results_.find_all("span", class_="data-header__club")[0]
league = trs_.find_all("")


#[-1].get('title') #### הבעיה פהההההההההה

print(trs)
print(type(trs))
print(len(trs))




print(0)




















champion_league = pd.read_csv(r'data_tables/ChampinLeagueFinalFixed.csv')

a = champion_league['home_team'].unique()

print(0)