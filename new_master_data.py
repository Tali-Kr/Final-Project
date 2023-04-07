import pandas as pd
from datetime import datetime, timedelta
from while_international_games import to_date, to_time, kot_determining, while_champion_european_leagues, \
    while_world_cup
from champion_relegated_methods import league_table
import time
# import Levenshtein
from fuzzywuzzy import fuzz
import numpy as np
import geopy.distance

t_start = time.time()  # To know the timerun of the program

# Show all the columns when print dataframe
pd.set_option("display.max_columns", None)

master_df = pd.read_csv(r'data_tables/ligathaal1221.csv')
master_df.rename(columns={'match_date': 'date', 'season_round': 'round'}, inplace=True)


def get_cities(team):
    """
    Extracts the city's name from the club's name.
    All of the clubs have their home city in the club's name, most of the time the name appears on the second word in the name.
    :param team: string. Team's name that needed to get the city's name.
    :return: string. City's name
    """
    club_city_name = team.split(' ', 1)  # Splits the team's name to seperate the city's name.
    if len(club_city_name) == 1:  # Addressing the option that a club may have only a one-word name.
        return str(club_city_name)
    else:
        return str(club_city_name[1])


def is_derby(record):
    """
    Checks if a game is a derby one. Meaning the function checks if the home team and the away team are from the same city.
    :param record: Series. A row from the dataframe that needs to check if the game is a derby.
    :return: 1 => if the game is a derby game.
             0 => if the game is NOT a derby game.
    """
    if get_cities(record['home_team']) == get_cities(record['away_team']):  # Checks the game if a city's derby.
        return 1
    else:  # The game isn't a derby game.
        return 0


def division(attendance, capacity):
    """
    Calculates the quotient of Attendance/Capacity.
    :param attendance: string. The amount of audience attanded in a game.
    :param capacity: string. The capacity of the stadium.
    :return: float. The quotient.
    """
    # Converting the strings into float values.
    attendance = float(str(attendance).replace(",", ""))
    capacity = float(str(capacity).replace(",", ""))
    return attendance / capacity


def lnFunc(record):
    """
    Calculating the ln of a given number.
    :param record: int. The number to calculate ln to.
    :return: The lan of the number.
    """
    if record != 0:
        return np.log(record)
    else:
        return 0


# Determining which team won or if the game ended with a draw.
master_df['home_won'] = master_df.apply(lambda x: 1 if x['home_score'] > x['away_score'] else 0, axis=1)
master_df['away_won'] = master_df.apply(lambda x: 1 if x['home_score'] < x['away_score'] else 0, axis=1)
master_df['draw'] = master_df.apply(lambda x: 1 if x['home_score'] == x['away_score'] else 0, axis=1)

master_df['derby'] = master_df.apply(is_derby, axis=1)  # Determining whether the game is a derby game.
master_df['date'] = master_df['date'].apply(to_date)  # Converting the column into datetime.datetime type.
master_df['kot'] = master_df['kot'].apply(to_time)  # Converting the column into datetime.datetime type.
master_df['kot'] = master_df['kot'].apply(lambda x: x + timedelta(hours=1))  # Adjusting to Israel time.
# Checking if the games were while the World Cup seasons
master_df['while_world_cup'] = master_df.apply(lambda x: while_world_cup(x['date']), axis=1)

# Creating a temp column to categorize the kot
master_df['kot_temp'] = master_df['kot'].apply(kot_determining)
# Converting the values in the temp column into one-hot encoded vectors in a temp dataframe
temp_df = pd.get_dummies(master_df.kot_temp, prefix='kot')
master_df = pd.concat([master_df, temp_df], axis=1, join='inner')  # Concating the temp df into the master df
master_df.drop(['kot_temp'], axis='columns', inplace=True)  # Dropping the temp column

# Checking if the games were while the "Champion Leauge"'s games.
master_df['while_champion_league'] = \
    master_df.apply(lambda x: while_champion_european_leagues(x['kot'], x['date'], "champion_league"), axis=1)
# Checking if the games were while top 5 european league's games.
master_df['while_european_games'] = \
    master_df.apply(lambda x: while_champion_european_leagues(x['kot'], x['date'], "earopean_leagues"), axis=1)

# Changing the tound type name to match to league_table's values.
master_df['round_type'] = master_df['round_type'].str.replace('Relegation round', 'Relegation Round')
master_df['round_type'] = master_df['round_type'].str.replace('Regular round', 'Regular Season')
master_df['round_type'] = master_df['round_type'].str.replace('Championship round', 'Championship Round')
master_df['round_type'] = master_df['round_type'].apply(lambda x: x[1:] if x == ' Championship Round' else x)

# Create 2 unique keys for every game (row). first for the home game and second for the away team.
master_df['home_key'] = master_df.apply(
    lambda x: str(x['season']) + ' - ' + str(x['round_type']) + ' - ' + str(x['round']) + ' - ' + str(x['home_team']),
    axis=1)
master_df['away_key'] = master_df.apply(
    lambda x: str(x['season']) + ' - ' + str(x['round_type']) + ' - ' + str(x['round']) + ' - ' + str(x['away_team']),
    axis=1)

# Joins the values of ['key', 'promoted', 'pts', 'team_pos', 'relegated', 'champion'] in league_table into
# the master_df by the 'key_home'
master_df = pd.merge(master_df, league_table[['key', 'promoted', 'pts', 'relegated', 'champion', 'pos_b4_game']],
                     how='left', left_on=['home_key'], right_on=['key'])
master_df.drop(['key'], axis='columns', inplace=True)
master_df.rename(columns={'promoted': 'home_promoted', 'pts': 'home_league_pts', 'relegated': 'home_is_relegated',
                          'champion': 'home_is_champion', 'pos_b4_game': 'home_pos_b4_game'}, inplace=True)

# Joins the values of ['key', 'promoted', 'pts', 'team_pos', 'relegated', 'champion'] in league_table into
# the master_df by the 'key_home'
master_df = pd.merge(master_df, league_table[['key', 'promoted', 'pts', 'relegated', 'champion', 'pos_b4_game']],
                     how='left', left_on=['away_key'], right_on=['key'])
master_df.drop(['key'], axis='columns', inplace=True)
master_df.rename(columns={'promoted': 'away_promoted', 'pts': 'away_league_pts', 'relegated': 'away_is_relegated',
                          'champion': 'away_is_champion', 'pos_b4_game': 'away_pos_b4_game'}, inplace=True)


def underdog(x):
    pre_season = int(x['season']) - 1
    home = x['home_team']
    away = x['away_team']
    prevois_season_bottom = master_df[(master_df['season'] == pre_season) &
                                      ((master_df['round_type'] == 'Relegation Round') & (master_df['round'] == 7))]
    prevois_season_top = master_df[(master_df['season'] == pre_season) &
                                   ((master_df['round_type'] == 'Championship Round') & (master_df['round'] == 10))]
    if (x['round_type'] == 'Regular Season') and (x['round'] == 1):
        if int(x['season']) > 2012:
            if home in prevois_season_top['home_team'].values or home in prevois_season_top['away_team'].values:
                if away in prevois_season_top['home_team'].values or away in prevois_season_top['away_team'].values:
                    if prevois_season_top[prevois_season_top['home_team'] == home].empty:
                        home_position = prevois_season_top[prevois_season_top['away_team'] == home]['away_position']
                    else:
                        home_position = prevois_season_top[prevois_season_top['home_team'] == home]['home_position']
                    if prevois_season_top[prevois_season_top['home_team'] == away].empty:
                        away_position = prevois_season_top[prevois_season_top['away_team'] == away]['away_position']
                    else:
                        away_position = prevois_season_top[prevois_season_top['home_team'] == away]['home_position']
                    if home_position.values > away_position.values:
                        return home
                    else:
                        return away
                else:
                    return away
            elif home in prevois_season_bottom['home_team'].values or home in prevois_season_bottom['away_team'].values:
                if away in prevois_season_bottom['home_team'].values or \
                   away in prevois_season_bottom['away_team'].values:
                    if prevois_season_bottom[prevois_season_bottom['home_team'] == home].empty:
                        home_position = prevois_season_bottom[prevois_season_bottom['away_team'] == home]['away_position']
                    else:
                        home_position = prevois_season_bottom[prevois_season_bottom['home_team'] == home]['home_position']
                    if prevois_season_bottom[prevois_season_bottom['home_team'] == away].empty:
                        away_position = prevois_season_bottom[prevois_season_bottom['away_team'] == away]['away_position']
                    else:
                        away_position = prevois_season_bottom[prevois_season_bottom['home_team'] == away]['home_position']

                    if home_position.values > away_position.values:
                        return home
                    else:
                        return away
                else:
                    if away in prevois_season_top['home_team'].values or away in prevois_season_top['away_team'].values:
                        return home
                    else:
                        return away
            else:
                if away in prevois_season_top['home_team'].values or away in prevois_season_top['away_team'].values:
                    return home
                elif away in prevois_season_bottom['home_team'].values or \
                     away in prevois_season_bottom['away_team'].values:
                    return home
                else:
                    return None
        return None
    else:
        if x['home_pos_b4_game'] > x['away_pos_b4_game']:
            return home
        else:
            return away


master_df['underdog'] = master_df.apply(underdog, axis=1)

################################
################################

# Import files
stadiums = pd.read_csv(r'data_tables/Stadiums_In_Israel.csv')
clubs_stadiums = pd.read_csv(r'data_tables/clubs_home_stadium_by_year.csv')

# Rename "Akko" to "Arce"
master_df['stadium'] = master_df['stadium'].str.replace('Akko', 'Arce')

# Create new data frames
df = pd.DataFrame()
df2 = pd.DataFrame()

# fill the two DataFrames with the stadiums in both files imported
df2['stadium'] = stadiums['Stadium']
df['stadium_name'] = master_df['stadium']

# Remove all the duplicates to see the unique name of the stadiums
df.drop_duplicates(subset=['stadium_name'], inplace=True)

# Remove NaNs
df.dropna(inplace=True)

# perform cross join that returns the Cartesian product of rows from the tables in the join -
# (combines each row from the first table with each row from the second table)
df = df.merge(df2, how='cross')

# Compare two string in each row and returns the partial ratio similarity between two strings
df['score'] = df[['stadium_name', 'stadium']].apply(lambda x: fuzz.partial_ratio(*x), axis=1)

# Sort the scores from large to small (Descending) to see who gets the higher ratio
df = df.sort_values(by='score', ascending=False)

# Removes the duplicates from the 'stadium_name' and keeps the first ones with the highest score
df.drop_duplicates(subset=['stadium_name'], inplace=True, keep='first')

# Reset the indexes
df = df.reset_index(drop=True)

# Join two tables to correct the names of the stadiums
master_df = master_df.merge(df[['stadium_name', 'stadium']], how='left', left_on=['stadium'], right_on=['stadium_name'])

master_df['stadium_x'] = master_df['stadium_y']
master_df.drop(['stadium_name', 'stadium_y'], axis='columns', inplace=True)
master_df.rename(columns={'stadium_x': 'stadium'}, inplace=True)

# Turn KOT from am/pm to 24 hours
master_df['kot'] = pd.to_datetime(master_df['kot']).dt.strftime('%H:%M')

# Create unique key for clubs stadiums
master_df['unique_home'] = master_df['home_team'] + master_df['season'].astype(str)
master_df['unique_away'] = master_df['away_team'] + master_df['season'].astype(str)
clubs_stadiums['unique'] = clubs_stadiums['club'] + clubs_stadiums['year'].astype(str)

# get home team by joining the clubs stadiums and year table with the dataset
master_df = master_df.merge(clubs_stadiums[['unique', 'stadium']], how='left', left_on=['unique_home'],
                            right_on=['unique'])
master_df.drop(['unique', 'unique_home'], axis='columns', inplace=True)
master_df.rename(columns={'stadium_y': 'home_team_stadium', 'stadium_x': 'game_stadium'}, inplace=True)

# get away team by joining the clubs stadiums and year table with the dataset
master_df = master_df.merge(clubs_stadiums[['unique', 'stadium']], how='left', left_on=['unique_away'],
                            right_on=['unique'])
master_df.drop(['unique', 'unique_away'], axis='columns', inplace=True)
master_df.rename(columns={'stadium': 'away_team_stadium'}, inplace=True)

# get home team stadium coordinates by joining the 'Stadium_in_israel' table and the dataset
master_df = master_df.merge(stadiums[['Stadium', 'coordinates']], how='left', left_on=['home_team_stadium'],
                            right_on=['Stadium'])
master_df.drop(['Stadium'], axis='columns', inplace=True)
master_df.rename(columns={'coordinates': 'home_coordinates'}, inplace=True)

# get away team stadium coordinates by joining the 'Stadium_in_israel' table and the dataset
master_df = master_df.merge(stadiums[['Stadium', 'coordinates']], how='left', left_on=['away_team_stadium'],
                            right_on=['Stadium'])
master_df.drop(['Stadium'], axis='columns', inplace=True)
master_df.rename(columns={'coordinates': 'away_coordinates'}, inplace=True)

# Distance between two stadiums
master_df['distance_in_km'] = master_df.apply(
    lambda x: geopy.distance.geodesic(x['home_coordinates'], x['away_coordinates']).km, axis=1)
master_df.drop(['home_coordinates', 'away_coordinates'], axis='columns', inplace=True)

# Get stadium capacity by joining 'Stadiums_in_israel' table and the dataset
master_df = master_df.merge(stadiums[['Stadium', 'Capacity']], how='left', left_on=['game_stadium'],
                            right_on=['Stadium'])
master_df.drop(['Stadium'], axis='columns', inplace=True)
master_df.rename(columns={'Capacity': 'capacity'}, inplace=True)
master_df['capacity'] = master_df['capacity'].astype(float)

# attendance ratio
master_df['attendance'] = master_df['attendance'].str.replace(",", "").astype(float)
master_df['att_ratio'] = master_df.apply(lambda x: division(x['attendance'], x['capacity']), axis=1)

# ln(attendance)
master_df['ln(attendance)'] = master_df.apply(lambda x: lnFunc(x['attendance']), axis=1)

# ln(capacity)
master_df['ln(capacity)'] = np.log(master_df['capacity'])

# Year in which the stadium were built
master_df = master_df.merge(stadiums[['Stadium', 'built_in']], how='left', left_on=['game_stadium'],
                            right_on=['Stadium'])
master_df['built_in'] = master_df['built_in'].astype(int)
master_df.drop(['Stadium'], axis='columns', inplace=True)

# Stadium's Age at the same time when the game occurred.
master_df['stadium_age'] = pd.DatetimeIndex(master_df['date'], dayfirst=True).year - master_df['built_in']
# Stadium's Age at the same time when the game occurred squared.
master_df['stadium_age_squared'] = master_df['stadium_age'].apply(lambda x: x ** 2)

# Home team city (join with 'Stadiums_in_israel' table and the dataset)
master_df = master_df.merge(stadiums[['Stadium', 'City']], how='left', left_on=['home_team_stadium'],
                            right_on=['Stadium'])
master_df.drop(['Stadium'], axis='columns', inplace=True)
master_df.rename(columns={'City': 'home_team_city'}, inplace=True)

# away team city (join with 'Stadiums_in_israel' table and the dataset)
master_df = master_df.merge(stadiums[['Stadium', 'City']], how='left', left_on=['away_team_stadium'],
                            right_on=['Stadium'])
master_df.drop(['Stadium'], axis='columns', inplace=True)
master_df.rename(columns={'City': 'away_team_city'}, inplace=True)

# Rearanging columns order for more logic order.
rearanged_cols_order = pd.read_csv('cols_name_new.csv')  # New order of the columns.
cols = rearanged_cols_order.columns.tolist()
master_df = master_df[cols]

# Determining the dat of the week of the game
master_df['day_of_week_num'] = master_df['date'].apply(lambda x: str(x.weekday()))  # Getting the day of the week
days_dic = {'0': 'Monday', '1': 'Tuesday', '2': 'Wednesday', '3': 'Thursday', '4': 'Friday', '5': 'Saturday',
            '6': 'Sunday'}  # Creating a dictionery to change the number of the day of the week to the name of the day.
master_df['day_of_week'] = master_df['day_of_week_num'].apply(lambda x: days_dic.get(x))

# To see what is the runtime
t_end = time.time()
print(t_end - t_start)

# master_df.to_csv("master_data__new__07_04.csv")
print(0)