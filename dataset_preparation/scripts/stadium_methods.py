import pandas as pd
import numpy as np
import geopy.distance
from fuzzywuzzy import fuzz

stadiums = pd.read_csv('../dt_prep_tables/Stadiums_In_Israel.csv')
clubs_stadiums = pd.read_csv('../dt_prep_tables/clubs_home_stadium_by_year.csv')


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


def stadium_related_dt_point(master_df):

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
    master_df = master_df.merge(clubs_stadiums[['unique', 'stadium','clubs_city']], how='left', left_on=['unique_home'], right_on=['unique'])
    master_df.drop(['unique','unique_home'], axis='columns', inplace=True)
    master_df.rename(columns={'stadium_y': 'home_team_stadium', 'stadium_x': 'game_stadium', 'clubs_city': 'home_team_city'}, inplace=True)

    # get away team by joining the clubs stadiums and year table with the dataset
    master_df = master_df.merge(clubs_stadiums[['unique', 'stadium','clubs_city']], how='left', left_on=['unique_away'], right_on=['unique'])
    master_df.drop(['unique','unique_away'], axis='columns', inplace=True)
    master_df.rename(columns={'stadium': 'away_team_stadium', 'clubs_city': 'away_team_city'}, inplace=True)

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

    return master_df
