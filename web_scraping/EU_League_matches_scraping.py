from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
from csv import writer
from tqdm import tqdm

def removeLinks(arr, str):
    del arr[:arr.index(str)]
    return arr

headers = {'User-Agent':
               'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}

games_in_season_links = pd.read_csv('season_games_links.csv')['Links'].tolist()
removeLinks(games_in_season_links,'https://www.transfermarkt.com/olympique-lyon_ea-guingamp/index/spielbericht/3047765')


with open('All_leagues6.csv', 'w', encoding='utf8', newline='') as f:  # for each year will be created a csv file
    theWriter = writer(f)
    header = ['match_date', 'kot', 'round_type', 'season_round', 'home_team', 'away_team', 'home_position', 'away_position', 'underdog', 'home_score', 'away_score', 'referee', 'stadium', 'attendance', 'season']  # the headers of the file will be
    theWriter.writerow(header)

    for linkz in tqdm(games_in_season_links):
        page = linkz
        pageTree = requests.get(page, headers=headers)
        pageSoup = BeautifulSoup(pageTree.content, 'html.parser')


        temp_array = []
        home_team = ""  # Done
        away_team = ""  # Done
        home_score = ""  # Done
        away_score = ""  # Done
        home_position = ""  # Done
        away_position = ""  # Done
        stadium = ""  # Done
        attendance = ""  # Done
        season_round = ""  # Done
        kot = ""  # Done
        match_date = ""  # Done
        referee = ""  # Done
        underdog = ""  # Done
        round_type = ""  # Done
        season = ""  # Done
        # Get the names of the teams
        for father in pageSoup.find_all("a", class_="sb-vereinslink"):
            temp_array.append(father.get('title'))
        try:
            home_team = temp_array[0]
        except Exception:
            home_team = None

        try:
            away_team = temp_array[1]
        except Exception:
            away_team = None
        temp_array = []

        # get the scores of the teams
        try:
            home_score = (pageSoup.find("div", class_="sb-endstand").text.strip().split("(")[0].split(":")[0])
        except Exception:
            home_score = None
        try:
            away_score = (pageSoup.find("div", class_="sb-endstand").text.strip().split("(")[0].split(":")[1])
        except Exception:
            away_score = None


        # get home and away position
        for father in pageSoup.find_all("p", rel="tooltip"):
            temp_array.append(father.text.split(":")[1].strip())
        try:
            home_position = temp_array[0]
        except Exception:
            home_position = None

        try:
            away_position = temp_array[1]
        except Exception:
            away_position = None
        temp_array = []

        try:
            # get the stadium name and attendance
            for father in pageSoup.find("span", class_="hide-for-small"):
                if father.text.strip() != "|" and father.text.strip() != "":
                    temp_array.append(father.text.strip())
        except Exception:
            print('failed')

        try:
            stadium = temp_array[0]
        except Exception:
            stadium = None
        try:
            attendance = temp_array[1].split(":")[1].strip().replace(".", "")
        except Exception:
            attendance = 0
        temp_array = []

        try:
            # get the round number & match date & KOT
            for father in pageSoup.find("p", class_="sb-datum hide-for-small"):
                if father.text.strip() != "|" and father.text.strip() != "":
                    temp_array.append(father.text.strip().replace("|  ", ""))
        except Exception:
            print('failed')

        try:
            season_round = temp_array[0].split(".")[0]
        except Exception:
            season_round = None
        try:
            match_date = (temp_array[1].split(",")[1].strip().split("/")[1])+"/"+(temp_array[1].split(",")[1].strip().split("/")[0])+"/"+(temp_array[1].split(",")[1].strip().split("/")[2])
        except Exception:
            match_date = None

        try:
            kot = temp_array[2]
        except Exception:
            kot = None
        temp_array = []

        try:
            # get referee
            for father in pageSoup.find("p", class_="sb-zusatzinfos"):
                if father.text.strip() != "":
                    temp_array.append(father.text.strip())
        except Exception:
            print('failed')
        try:
            referee = temp_array[2]
        except Exception:
            referee = None
        temp_array = []

        # determine who is the underdog
        try:
            if int(home_position) < int(away_position):
                underdog = away_team
            else:
                underdog = home_team
        except Exception:
            print('failed')

        try:
            round_type = pageSoup.find("h2").text.strip().split(" - ")[1]
        except Exception:
            round_type = "Regular round"

        # Get the season
        try:
            for g1 in pageSoup.find_all("p", class_='sb-datum hide-for-small'):
                season = int(g1.find("a").get('href').split("/")[6])
        except Exception:
            season = None

        # Test link
        # print("Home team:" + home_team + "\n" +
        #       "Away team:" + away_team + "\n" +
        #       "Home Score:" + home_score + "\n" +
        #       "Away Score:" + away_score +"\n" +
        #       "Home Position:" + home_position +"\n" +
        #       "Away Position:" + away_position +"\n" +
        #       "Stadium:" + stadium +"\n" +
        #       "Attendance:" + str(attendance) +"\n" +
        #       "Season Round:" + season_round +"\n" +
        #       "KOT:" + kot +"\n" +
        #       "Match Date:" + match_date +"\n" +
        #       "Referee:" + referee +"\n" +
        #       "Underdog:" + underdog +"\n" +
        #       "Round Type:" + round_type)
        info = [match_date, kot, round_type, season_round, home_team, away_team, home_position, away_position, underdog, home_score, away_score, referee, stadium, attendance, season]
        theWriter.writerow(info)
print("Done! - 100%")
