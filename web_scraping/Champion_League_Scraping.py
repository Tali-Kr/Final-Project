from bs4 import BeautifulSoup
import requests
from csv import writer
from tqdm import tqdm

# set user agent in HTTP headers for web scraping that mimic a particular web browser
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/' +
                         '537.36 (KHTML, like Gecko) Chrome/' +
                         '47.0.2526.106 Safari/537.36'
           }

# for each year will be created a csv file
with open('../dataset_preparation/dt_prep_tables/champion_league_2012_21.csv', 'w', encoding='utf8', newline='') as f:
    theWriter = writer(f)
    header = ['match_date', 'kot', 'round_type', 'season_round', 'home_team', 'away_team', 'home_score', 'away_score',
              'referee', 'stadium', 'attendance', 'season']  # the headers of the file will be
    theWriter.writerow(header)

# Determine the scraping years which also be will be added to the URL links
    fromYear = 2012
    toYear = 2021
# Determine the groups which will be added to the URL links
    groups = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'AFH', 'AFR', 'VFH', 'VFR', 'HFH', 'HFR', 'FF']
    group_year_links = []
    match_links = []
# Appends all the Champion League url links into a list
    for seasonYear in tqdm(range(fromYear, toYear+1)):
        for group in groups:
            group_year_links.append('https://www.transfermarkt.com/' +
                                    'uefa-champions-league/' +
                                    'spieltag/pokalwettbewerb/CL/plus/0?saison_id=' +
                                    str(seasonYear) + '&gruppe=' + group)
# Go through all the links of the leagues and get the game links and append them into a list called "match_links"
    for link in tqdm(group_year_links):
        page = link
        pageTree = requests.get(page, headers=headers)
        pageSoup = BeautifulSoup(pageTree.content, 'html.parser')

        for i in pageSoup.find_all("a", class_='liveLink'):
            match_links.append('https://www.transfermarkt.com'+i.get('href'))
# Go through all the links of games and get all the data
    for linkz in tqdm(match_links):
        page = linkz
        pageTree = requests.get(page, headers=headers)
        pageSoup = BeautifulSoup(pageTree.content, 'html.parser')

        # running check
        print(linkz)
        print(str(round((percent_check / len(match_links)) * 100, 2)) + "%")
        percent_check = percent_check + 1

        temp_array = []
        home_team = ""  # Done
        away_team = ""  # Done
        home_score = ""  # Done
        away_score = ""  # Done
        stadium = ""  # Done
        attendance = ""  # Done
        season_round = ""  # Done
        kot = ""  # Done
        match_date = ""  # Done
        referee = ""  # Done
        round_type = ""  # Done
        season = ""  # Done

        # Get the names of the teams
        for father in pageSoup.find_all("a", class_="sb-vereinslink"):
            temp_array.append(father.get('title'))
        try:
            home_team = temp_array[0]
        except IndexError:
            home_team = None

        try:
            away_team = temp_array[1]
        except IndexError:
            away_team = None
        temp_array = []

        # get the scores of the teams
        try:
            home_score = (pageSoup.find("div", class_="sb-endstand").text.strip().split("(")[0].split(":")[0])
        except IndexError:
            home_score = None
        try:
            away_score = (pageSoup.find("div", class_="sb-endstand").text.strip().split("(")[0].split(":")[1])
        except IndexError:
            away_score = None

        # # get home and away position after the game
        # for father in pageSoup.find_all("p", rel="tooltip"):
        #     temp_array.append(father.text.split(":")[1].strip())
        # try:
        #     home_position = temp_array[0]
        # except IndexError:
        #     home_position = None
        #
        # try:
        #     away_position = temp_array[1]
        # except:
        #     away_position = None
        # temp_array = []

        # get the stadium name and attendance
        for father in pageSoup.find("span", class_="hide-for-small"):
            if father.text.strip() != "|" and father.text.strip() != "":
                temp_array.append(father.text.strip())

        try:
            stadium = temp_array[0]
        except IndexError:
            stadium = None
        try:
            attendance = temp_array[1].split(":")[1].strip().replace(".", "")
        except IndexError:
            attendance = 0
        temp_array = []

        # get the round number & match date & KOT
        for father in pageSoup.find("p", class_="sb-datum hide-for-small"):
            if father.text.strip() != "|" and father.text.strip() != "":
                temp_array.append(father.text.strip().replace("|  ", ""))

        season_round = temp_array[0].split(".")[0]
        match_date = (temp_array[1].split(",")[1].strip().split("/")[1]) + "/" + (
        temp_array[1].split(",")[1].strip().split("/")[0]) + "/" + (
                     temp_array[1].split(",")[1].strip().split("/")[2])
        try:
            kot = temp_array[2]
        except IndexError:
            kot = None
        temp_array = []

        # get referee
        for father in pageSoup.find("p", class_="sb-zusatzinfos"):
            if father.text.strip() != "":
                temp_array.append(father.text.strip())
        try:
            referee = temp_array[2]
        except IndexError:
            referee = None
        temp_array = []

        try:
            round_type = pageSoup.find("h2").text.strip().split(" - ")[1]
        except IndexError:
            round_type = "Regular round"

        # Get the season
        try:
            for g1 in pageSoup.find_all("a", class_='direct-headline__link'):
                season = g1.get('href').split("/")[6]
        except IndexError:
            season = None

        info = [match_date, kot, round_type, season_round, home_team, away_team,
                home_score, away_score, referee, stadium, attendance, season]
        theWriter.writerow(info)
print("Done! - 100%")
