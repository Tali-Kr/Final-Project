from datetime import datetime
import pandas as pd
import numpy as np
import pytz
from champion_relegated_methods import *
from while_international_games import *
from archived_european_vs_israeli_games import *

master_df = pd.read_csv(r'data_tables/ligathaal1221.csv')


def split_club_name(club_name):  # Splits the clubs name to prep for the get_cities function.
    return club_name.split(' ', 1)


def get_cities(club_name):
    """
    Extracts the city's name from the club's name.
    All of the clubs have their home city in the club's name, most of the time the name appears on the second word in the name.
    :param club_name: Series. A column from the dataframe that needs to extract cities names.
    :return: string. City's name
    """
    club_name = split_club_name(club_name)
    if len(club_name) == 1:  # Addressing the option that a club may have only a one-word name.
        return str(club_name)
    else:
        return str(club_name[1])


def is_derby(record):
    """
    Checks if a game is a derby one. Meaning the function checks if the home team and the away team are the same city.
    :param record: Series. A row from the dataframe that needs to check if the game is a derby.
    :return: 1 => if the game is a derby game.
             0 => if the game is NOT a derby game.
    """
    if get_cities(record['homeTeamName']) == get_cities(record['awayTeamName']):  # Checks the game if a city's derby.
        return 1
    else:  # The game isn't a derby game.
        return 0


master_df['derby'] = master_df.apply(is_derby, axis=1)
master_df['date'] = master_df.apply(lambda x: timestamp_to_date(x['timestamp']), axis=1)
master_df['KOT'] = master_df.apply(lambda x: timestamps_to_time(x['timestamp']), axis=1)
master_df['while_world_cup'] = master_df.apply(lambda x: while_world_cup(x['date']), axis=1)

hours_names = ['15', '16', '17', '18', '19', '20', '21', '22']
# Will help later to define the names of the new columns and to determinate the KOT group.
start_t = ['15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00']
end_t = ['16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00']
form = '%H:%M'  # Wanted format for the time defenition.

for t in range(len(start_t)):
    # Converting the type of the list from string list to time list
    start_t[t] = datetime.strptime(start_t[t], form).time()
    # Converting the type of the list from string list to time list
    end_t[t] = datetime.strptime(end_t[t], form).time()


def determining_KOT(t, start, end):
    """
    Checking the KOT group.
    :param start: time. Lower bound for the tested category.
    :param end: time. Upper bound for the tested category.
    :param t: time. The KOT of the tested match.
    :return: if the KOT of the tested match fit in between the bounds => the tested catecory is matching - returning 1
                                                                                         if not matching - returning 0.
    """
    if start <= t < end:
        return 1
    else:
        return 0


# Creating 7 new coulmns to note the KOT group (kick of time = start of the match)
for t in range(len(end_t)):
    col = 'KOT' + '_' + hours_names[t] + '_' + hours_names[t + 1]
    master_df[col] = master_df['KOT'].apply(determining_KOT, args=(start_t[t], end_t[t],))

# Creating new dataframe for league tables csv file
league_tables = pd.read_csv(r'data_tables/Tables_Final.csv')

seasons = [2016, 2017, 2018, 2019, 2020, 2021]
seasons_wrong = ['16/17', '17/18', '18/19', '19/20', '20/21', '21/22']


def value_matching(saving_to_df, saving_to_col, value, right_value):
    saving_to_df.loc[saving_to_df[saving_to_col] == value, saving_to_col] = right_value


for s in range(len(seasons)):  # Matching the season values to the values in master_df
    value_matching(league_tables, 'season', seasons_wrong[s], seasons[s])

# Matching the clubs name values to the values in master_df
league_t_df_club_names = list(set(league_tables['clubs_name']))
master_df_club_names = list(set(master_df['homeTeamName']))


def differnts_in_list(l1, l2):
    return list(set(l1) - set(l2))


differnt_club_names = sorted(differnts_in_list(master_df_club_names, league_t_df_club_names))
wrong_club_names = ['FC Ashdod', 'Ihud Bnei Sakhnin', 'Bnei Yehuda Tel Aviv', 'Hapoel Jerusalem', 'Hapoel Nof HaGalil',
                    'Hapoel Raanana', 'Sekzia Ness Ziona']

for n in range(len(differnt_club_names)):
    value_matching(league_tables, 'clubs_name', wrong_club_names[n], differnt_club_names[n])

## changing the pos from 0-13 / 0-5 / 0-7 to 1-14 / 1-6 / 1-8
league_tables['team_pos'] = league_tables['team_pos'].apply(lambda pos: pos + 1)

## Changing the 'game' name
value_matching(league_tables, 'game', 'Relegation round', 'Relegation Round')
value_matching(league_tables, 'game', 'Regular', 'Regular Season')


## Creates unique key for league_table dataframe
def key_maker_league_tables(record):
    return str(record['game']) + ' - ' + str(record['round']) + ' - ' + str(record['season']) + ' - ' + \
        str(record['clubs_name'])


league_tables['key'] = league_tables.apply(key_maker_league_tables, axis=1)


# Creates unique key for master_df dataframe
def key_maker_master_df(record, name):
    return str(record['round']) + ' - ' + str(record['season']) + ' - ' + str(record[name])


# Create 2 unique keys for every game (row). first for the home game and second for the away team.
master_df['key_home'] = master_df.apply(key_maker_master_df, args=('homeTeamName',), axis=1)
master_df['key_away'] = master_df.apply(key_maker_master_df, args=('awayTeamName',), axis=1)

# Check if the teams played as a champion or relegated.
league_tables['relegated'] = league_tables.apply(
    lambda x: relegation_championship_check(x) if x['game'] == 'Relegation Round' else None, axis=1)
league_tables['champion'] = league_tables.apply(
    lambda x: relegation_championship_check(x) if x['game'] == 'Championship Round' else None, axis=1)
print(0)
# Joins the values of ['key', 'promoted', 'pts', 'team_pos', 'relegated', 'champion'] in league_table into
# the master_df by the 'key_home'
master_df = pd.merge(master_df, league_tables[['key', 'promoted', 'pts', 'team_pos', 'relegated', 'champion']],
                     how='left', left_on=['key_home'], right_on=['key'])
master_df.drop(['key'], axis='columns', inplace=True)
master_df.rename(columns={'promoted': 'home_promoted', 'pts': 'home_league_pts', 'team_pos': 'home_pos',
                          'relegated': 'home_is_relegated', 'champion': 'home_is_champion'}, inplace=True)

# Joins the values of ['key', 'promoted', 'pts', 'team_pos', 'relegated', 'champion'] in league_table into
# the master_df by the 'away_home'
master_df = pd.merge(master_df, league_tables[['key', 'promoted', 'pts', 'team_pos', 'relegated', 'champion']],
                     how='left', left_on=['key_away'], right_on=['key'])
master_df.drop(['key'], axis='columns', inplace=True)
master_df.rename(columns={'promoted': 'away_promoted', 'pts': 'away_league_pts', 'team_pos': 'away_pos',
                          'relegated': 'away_is_relegated', 'champion': 'away_is_champion'}, inplace=True)

# Checks if the game was mwanwhile a european game.
master_df = master_df.apply(meanwhile_european_game, axis=1)

# master_df.to_csv("master_data.csv")  # Creating new csv file that contains all the csv season files.
print("END")
