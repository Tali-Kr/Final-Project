from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np

# Get all the needed links for the web scraping
# What league and how many round in each league
leagues = [['ligat-haal','ISR1', '1', '26'], ['premier-league', 'GB1', '1', '38'], ['laliga', 'ES1', '1', '38'],
           ['bundesliga', 'L1', '1', '34'], ['ligue-1', 'FR1', '1', '38'], ['serie-a', 'IT1', '1', '38']]
league_links = []   # League links in from 'fromYear' to 'toYear'
season_links = []   # All the links to the games in the leagues from 'fromYear' to 'toYear'
fromYear = '2012'   # From which year
toYear = '2021'     # To which year
# Get all the League links in from 'fromYear' to 'toYear'
for league in leagues[0]:
    for year in range(int(fromYear), int(toYear)+1):
        league_links.append('https://www.transfermarkt.com/' + league[0] +
                            '/gesamtspielplan/wettbewerb/' + league[1] +
                            '?saison_id=' + str(year) +
                            "&spieltagVon=" + league[2] + '&spieltagBis=' + league[3])

# set user agent in HTTP headers for web scraping that mimic a particular web browser
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/' +
                         '537.36 (KHTML, like Gecko) Chrome/' +
                         '47.0.2526.106 Safari/537.36'
           }

# Get all the links to the games in the leagues from 'fromYear' to 'toYear'
for league_link in league_links:
    page = league_link
    pageTree = requests.get(page, headers=headers)
    pageSoup = BeautifulSoup(pageTree.content, 'html.parser')

    for season_link in pageSoup.find_all("a", class_="ergebnis-link"):
        season_links.append('https://www.transfermarkt.com' + season_link.get('href'))
# Put all the links in a numpy array
season_links = np.array(season_links)

# Put the array in dataframe
df = pd.DataFrame(season_links, columns=['Links'])

# Export dataframe to a CSV file
df.to_csv('season_games_links.csv', index=False)
