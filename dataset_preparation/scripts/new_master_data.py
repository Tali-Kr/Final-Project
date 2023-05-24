import pandas as pd
from datetime import timedelta
import stadium_methods
from while_international_games import to_date, to_time, kot_determining, while_champion_european_leagues, \
    while_world_cup
import champion_relegated_methods
import time
# import Levenshtein

t_start = time.time()  # To know the timerun of the program

# Show all the columns when print dataframe
pd.set_option("display.max_columns", None)

master_df = pd.read_csv('../dt_prep_tables/ligathaal1221.csv')
# master_df = pd.read_csv('../dt_prep_tables/New_data_1005_1902.csv')  # Relevant for only new data
master_df.rename(columns={'match_date': 'date', 'season_round': 'round'}, inplace=True)


def get_cities(team):
    """
    Extracts the city's name from the club's name.
    All of the clubs have their home city in the club's name.
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
    Checks if the game is a derby. Meaning the function checks if the home and away teams are from the same city.
    :param record: Series. A row from the dataframe that needs to check if the game is a derby.
    :return: 1 => if the game is a derby game.
             0 => if the game is NOT a derby game.
    """
    if get_cities(record['home_team']) == get_cities(record['away_team']):  # Checks the game if a city's derby.
        return 1
    else:  # The game isn't a derby game.
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
# master_df['kot_22_23'] = False    # Relevant for only new data

# Checking if the games were while the "Champion Leauge"'s games.
master_df['while_champion_league'] = \
    master_df.apply(lambda x: while_champion_european_leagues(x['kot'], x['date'], "champion_league"), axis=1)
# Checking if the games were while top 5 european league's games.
master_df['while_european_games'] = \
    master_df.apply(lambda x: while_champion_european_leagues(x['kot'], x['date'], "earopean_leagues"), axis=1)

# Changing the round type name to match league_table's values.
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

# Getting the league tables df from the champion_relegated_methods.py file
league_table = champion_relegated_methods.get_league_table()

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

# master_df.to_csv('../dt_prep_tables/master_data_temp_test_21_05.csv')  # Saves the original data set into csv file.
print("new_master_data.py  :  master_df.to_csv('master_data_temp_test_21_05.csv')  -  DONE")
# master_df.to_csv('../dt_prep_tables/master_new_data_temp_1205.csv')  # Saves the new data set into csv file. Relevant for only new data

from is_underdog_check import is_underdog, pts_b4_game_home, pts_b4_game_away

master_df['underdog'] = master_df.apply(is_underdog, axis=1)
print("new_master_data.py  :  master_df['underdog']  -  DONE")
master_df['home_lg_b4_game'] = master_df.apply(lambda x: pts_b4_game_home(x), axis=1)
master_df['away_lg_b4_game'] = master_df.apply(lambda x: pts_b4_game_away(x), axis=1)
print("new_master_data.py  :  ['away_lg_b4_game']  -  DONE")

master_df = stadium_methods.stadium_related_dt_point(master_df)
print()

# Determining the dat of the week of the game.
master_df['day_of_week_num'] = master_df['date'].apply(lambda x: str(x.weekday()))  # Getting the day of the week.
days_dic = {'0': 'Monday', '1': 'Tuesday', '2': 'Wednesday', '3': 'Thursday', '4': 'Friday', '5': 'Saturday',
            '6': 'Sunday'}  # Creating a dictionery to change the number of the day of the week to the name of the day.
master_df['day_of_week'] = master_df['day_of_week_num'].apply(lambda x: days_dic.get(x))

# Rearanging columns order for more logic order.
rearanged_cols_order = pd.read_csv('../dt_prep_tables/cols_name_new.csv')  # New order of the columns.
cols = rearanged_cols_order.columns.tolist()
master_df = master_df[cols]

# To see what is the runtime
t_end = time.time()
print(t_end - t_start)

# master_df.to_csv('../dt_prep_tables/master_data__new__08_05.csv')
# master_df.to_csv('../dt_prep_tables/master_new_data_12_05_1525.csv')  # Relevant for only new data
print(0)