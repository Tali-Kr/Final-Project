from bs4 import BeautifulSoup
import requests
from csv import writer
from tqdm import tqdm

match_links = []

link_regluar_saeson = 'https://www.transfermarkt.com/ligat-haal/gesamtspielplan/wettbewerb/ISR1?saison_id=2022&spieltagVon=1&spieltagBis=26'
link_championshiop =  'https://www.transfermarkt.com/ligat-haal-championship-round/gesamtspielplan/wettbewerb/ISRF/saison_id/2022'
link_relegation = 'https://www.transfermarkt.com/ligat-haal-relegation-round/gesamtspielplan/wettbewerb/ISRA/saison_id/2022'
links = [link_regluar_saeson, link_championshiop, link_relegation]

headers = {'User-Agent':
               'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}


# region extract data
# Get all the needed links for the web scraping
for link in links:
    page = link
    pageTree = requests.get(page, headers=headers)
    pageSoup = BeautifulSoup(pageTree.content, 'html.parser')

    for match_link in pageSoup.find_all("a", class_="ergebnis-link"):
        match_links.append('https://www.transfermarkt.com' + match_link.get('href'))

# Saving all the extracted data into csv file.
with open('../dataset_preparation/dt_prep_tables/ligat_haal_2022.csv', 'w', encoding='utf8', newline='') as f:
    theWriter = writer(f)
    # the headers of the file will be
    header = ['match_date', 'kot', 'round_type', 'season_round', 'home_team', 'away_team', 'home_position',
              'away_position', 'underdog', 'home_score', 'away_score', 'referee', 'stadium', 'attendance', 'season']
    theWriter.writerow(header)

    for link in tqdm(match_links):
        page = link
        pageTree = requests.get(page, headers=headers)
        pageSoup = BeautifulSoup(pageTree.content, 'html.parser')

        temp_list = []  # Sacews the extracted data point into a temp list.

        # Get home and away teams.
        for father in pageSoup.find_all("a", class_="sb-vereinslink"):
            temp_list.append(father.get('title'))
        home_team = temp_list[0]
        away_team = temp_list[1]
        temp_list = []  # reset the list

        # Get home and away teams' scores.
        home_score = (pageSoup.find("div", class_="sb-endstand").text.strip().split("(")[0].split(":")[0])
        away_score = (pageSoup.find("div", class_="sb-endstand").text.strip().split("(")[0].split(":")[1])

        # Get teams' position after the match.
        for father in pageSoup.find_all("p", rel="tooltip"):
            temp_list.append(father.text.split(":")[1].strip())
        home_position = temp_list[0]
        away_position = temp_list[1]
        temp_list = []  # reset the list

        # Get the stadium of the match, attandance in the match and the referee of the match.
        for father in pageSoup.find("p", class_="sb-zusatzinfos"):
            if father.text.strip() != "|" and father.text.strip() != "":
                temp_list.append(father.text)
        stadium = temp_list[0].split("|")[0].strip()
        try:
            attendance = int(temp_list[0].split("|")[1].split(":")[1].strip().replace('.', ""))
        except:
            attendance = 0
        referee = temp_list[2]
        temp_list = []  # reset the list

        # Get the match date and kot and the round of the match.
        for father in pageSoup.find("p", class_="sb-datum hide-for-small"):
            if father.text.strip() != "|" and father.text.strip() != "":
                temp_list.append(father.text.strip().replace("|  ", ""))
        season_round = temp_list[0].split('.')[0]
        match_date = ((temp_list[1].split(',')[1].strip().split('/')[1]) + "/" +
                     (temp_list[1].split(',')[1].strip().split('/')[0]) + "/" +
                     (temp_list[1].split(',')[1].strip().split('/')[2]))
        kot = temp_list[2]
        temp_list = []  # reset the list

        # Set witch team is the underdog
        if int(home_position) < int(away_position):
            underdog = away_team
        else:
            underdog = home_team

        # Get the round type of the match.
        try:
            round_type = pageSoup.find("h2").text.strip().split(" - ")[1]
        except:
            round_type = "Regular round"

        # Get the seson of the match.
        for g1 in pageSoup.find_all("p", class_='sb-datum hide-for-small'):
            season = int(g1.find("a").get('href').split("/")[6])

        info = [match_date, kot, round_type, season_round, home_team, away_team, home_position, away_position, underdog,
                home_score, away_score, referee, stadium, attendance, season]
        theWriter.writerow(info)
print("Done! - 100%")

#endregion
