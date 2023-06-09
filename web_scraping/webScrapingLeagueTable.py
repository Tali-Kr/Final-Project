from bs4 import BeautifulSoup
import requests
import pandas as pd, numpy as np
import re
from csv import writer

headers = {'User-Agent':
               'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}
fromYear = 2012
toYear = 2021
all_the_link = []
for year in range(int(fromYear), int(toYear) + 1):
    link = "https://www.transfermarkt.com/ligat-haal/formtabelle/wettbewerb/ISR1?saison_id=+" + str(
        year) + "&min=1&max="
    for round in range(1, 27):
        all_the_link.append(link + str(round))
    link = "https://www.transfermarkt.com/ligat-haal-meisterrunde/formtabelle/wettbewerb/ISRF?saison_id=" + str(
        year) + "&min=1&max="
    for round in range(1, 11):
        all_the_link.append(link + str(round))
    link = "https://www.transfermarkt.com/ligat-haal-abstiegsrunde/formtabelle/wettbewerb/ISRA?saison_id=" + str(
        year) + "&min=1&max="
    for round in range(1, 8):
        all_the_link.append(link + str(round))

with open('Tables.csv', 'w', encoding='utf8', newline='') as f:  #for each year will be created a csv file
    theWriter = writer(f)
    header = ['season', 'clubs_name','round' , 'promoted', 'game', 'win', 'draw', 'lose', 'goals', 'g_difference', 'pts', 'team_pos']  # the headers of the file will be
    theWriter.writerow(header)

    for link in all_the_link:

        page = link  # The link we are getting data from
        print(link)
        pageTree = requests.get(page, headers=headers)
        pageSoup = BeautifulSoup(pageTree.content, 'html.parser')

        try:
            season = pageSoup.find_all("option", selected="selected")[0].text
        except IndexError:
            season = None

        # define variables
        clubs_names = []
        promted = []  # team that has been promoted last season to current league 0 if wasn't promoted 1 if it was promoted
        match = []
        win = []
        draw = []
        lose = []
        goals = []
        g_difference = []  # goal difference
        pts = []
        team_pos = []

        # getting club's name and if it has been promoted from lower league
        for club in pageSoup.find_all("td", {'class': "no-border-links hauptlink"}):
            clubs_names.append((club.find("a")).get('title'))
            if club.find("span") != None:
                promted.append(1)
            else:
                promted.append(0)

        for count in range(len(clubs_names)):
            team_pos.append(count)

        # define the game type
        if len(clubs_names) == 14:
            game = "Regular"
        elif len(clubs_names) == 6:
            game = "Championship Round"
        else:
            game = "Relegation round"

        # get all the round's info
        data = []
        for children in pageSoup.find_all("tbody"):
            for child in children.find_all("td", class_="zentriert"):
                if child.text.strip() != "":
                    data.append(child.text.strip())

        # if there are 7 fields need range of 7 if there are 8 fields need range of 8
        if len(data) == 98 or len(data) == 42 or len(data) == 56:
            indx_range = 7
        else:
            indx_range = 8

        # enter all the round's  info of the team
        for i in range(len(clubs_names)):
            match.append(data[0])
            win.append(data[1])
            draw.append(data[2])
            lose.append(data[3])
            goals.append(data[4])
            g_difference.append(data[5])
            pts.append(data[6])
            del data[0:indx_range]

        for i in range(len(clubs_names)):
            info = [season, clubs_names[i], match[i], promted[i], game, win[i], draw[i], lose[i], goals[i], g_difference[i], pts[i], team_pos[i]]  # get everything in one list
            theWriter.writerow(info)  # write the row.
