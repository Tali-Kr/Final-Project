import pandas as pd
import re

# league_table = pd.read_csv(r'dataset_preparation/dt_prep_tables/league_tables_2012_2021.csv')
league_table = pd.read_csv(r'dataset_preparation/dt_prep_tables/league_tables_2021_2022_fixed.csv')
# league_table = pd.read_csv(r'dataset_preparation/dt_prep_tables/league_tables_new_2022_fixed_1205.csv')  # Relevant for only new data

#region done in the league_table_fixer
# Changing the 'game' name
# league_table['game'] = league_table['game'].str.replace('Relegation round', 'Relegation Round')
# league_table['game'] = league_table['game'].str.replace('Regular', 'Regular Season')
# # changing the pos from 0-13 / 0-5 / 0-7 to 1-14 / 1-6 / 1-8
# league_table['team_pos'] = league_table['team_pos'].apply(lambda pos: pos + 1)
# Creates unique key for each record in league_table dataframe
#endregion
print()
#region Relevant for only new data
# numeric_cols = league_table[['season', 'round', 'promoted', 'team_pos', 'pts', 'g_difference', 'win', 'draw', 'lose',
#                              'goals_for', 'goals_against']]
#
# for col in numeric_cols:
#     league_table[col] = league_table[col].astype('Int64')
#endregion

league_table['key'] = league_table.apply(lambda x: str(x['season']) + ' - ' + str(x['round_type']) + ' - ' + str(x['round'])
                                                   + ' - ' + str(x['team']), axis=1)
print("champion_relegated_methods.py  :  league_table['key']  -  DONE")


def previous_round_pos(record):
    """
    Getting team's position on the previous round.
    :param record: series.
    :return: int. the position on the previous round.
    """
    pos = 0  # Reset of the variable.
    # All the rounds but the first of each round type (regular season, relegation or championship round).
    if record['round'] != 1:
        # Fillters out the relevant season, round type, team's name in the previous round, and gets the position.
        pos = league_table[(league_table['round_type'] == record['round_type']) &
                           (league_table['season'] == record['season']) &
                           (league_table['round'] == (record['round']) - 1) &
                           (league_table['team'] == (record['team']))]['team_pos']
        # print(record['key'])
        return pos.values[0]
    else:  # For the first round of each round type (regular season, relegation or championship round).
        # The check is irrelevant for the first round in the regular season.
        if record['round_type'] == 'Relegation Round' or record['round_type'] == 'Championship Round':
            pos = league_table[(league_table['round_type'] == 'Regular Season') &
                               (league_table['season'] == record['season']) &
                               (league_table['round'] == 26) &
                               (league_table['team'] == (record['team']))]['team_pos']
            # print(record['key'])
            return pos.values[0]
    # print(record['key'])
    return pos


# Getting the teams's position in the beginning of the game.
league_table['pos_b4_game'] = league_table.apply(lambda x: previous_round_pos(x), axis=1)
print("champion_relegated_methods.py  :  league_table['pos_b4_game']  -  DONE")

def pts_difference(code, record):
    """
    Calculate the points difference between 6th  and 7th or 8th places or between 1st and 2nd places.
    :param code: int. Tells witch playoff is needes (7 => bottom playoff ; 10 => top playoff)
    :param record: series.
    :return: int. The difference bwtween places.
    """
    if code == 7:  # code = 7 => bottom playoff (relegation)
        # For bottom playoff the importent pts differences are between 6th and 7th / 8th places.
        # pos_6 is the number of pts that the 6th place team had on the previous round of the given record's.
        pos_6 = int(league_table[(league_table['round_type'] == record['round_type']) &
                                 (league_table['season'] == record['season']) &
                                 (league_table['round'] == (record['round']) - 1) &
                                 (league_table['team_pos'] == 6)]['pts'])

        # team_pos is the number of pts that the team had on the previous round of the given record's
        # (the team's places are 7th or 8th).
        team_pos = int(league_table[(league_table['round_type'] == record['round_type']) &
                                    (league_table['season'] == record['season']) &
                                    (league_table['round'] == (record['round']) - 1) &
                                    (league_table['team'] == record['team'])]['pts'])
        diff = (pos_6 - team_pos)
        return diff

    else:  # code = 10 => top playoff (championship)
        # For top playoff the importent pts difference is between 1st and 2nd places.
        # pos_1 is the number of pts that the team had on the previous round of the given record's that was in 1st place
        pos_1 = int(league_table[(league_table['round_type'] == record['round_type']) &
                                 (league_table['season'] == record['season']) &
                                 (league_table['round'] == (record['round']) - 1) &
                                 (league_table['team_pos'] == 1) &
                                 (league_table['team'] == record['team'])]['pts'])

        # pos_2 is the number of pts that the 2nd place team had on the previous round of the given record's.
        pos_2 = int(league_table[(league_table['round_type'] == record['round_type']) &
                                 (league_table['season'] == record['season']) &
                                 (league_table['round'] == (record['round']) - 1) &
                                 (league_table['team_pos'] == 2)]['pts'])
        diff = (pos_1 - pos_2)
        return diff


def relegation_championship_check(record):
    """
    Checks if in a given record one of the teams played as already champion (in the champion round) or
                                                played as already relegated (in the relegation round)
    :param record: series.
    :return: 1 => played as already champion - in the champion round or
                  played as already relegated - in the relegation round
             0 => Didnt play as .....
    """
    if record['round_type'] == 'Relegation Round':  # Checks if the game is in the relegation round.
        if record['round'] > 4:  # The relegation check is relevant from the 5th round.
            if record['team_pos'] > 6:  # The relegation check is relevant for the 7th and 8th places.
                max_pts = 3 * (7 - (record['round'] - 1))  # Max pts that the team can win until the end of the season.
                if pts_difference(7, record) > max_pts:
                    # If the pts difference between 6th place and the given team's place is bigger than the max pts that
                    # the team can win => the given team is for sure relegated.
                    return 1
                else:  # If NOT => It can't be said for sure if the given team is relegated.
                    return 0

    elif record['round_type'] == 'Championship Round':  # Checks if the game is in the championship round.
        if record['round'] > 5:  # The championship check is relevant from the 7th round.
            if record['team_pos'] == 1:  # The championship check is relevant only for the first place in the round.
                a = league_table[(league_table['round_type'] == record['round_type']) &
                                 (league_table['season'] == record['season']) &
                                 (league_table['round'] == (record['round']) - 1) &
                                 (league_table['team_pos'] == 1) &
                                 (league_table['team'] == record['team'])]
                if not a.empty:
                    max_pts = 3 * (10 - (
                            record['round'] - 1))  # Max pts that the team can win until the end of the season.
                    if pts_difference(10, record) > max_pts:
                        # If the max pts that can be win until the end of the season is bigger than the difference between
                        # 1st and 2nd places => the given team is the champion for sure.
                        return 1
                    else:  # If NOT => It can't be said for sure if the given team is champion.
                        return 0
                else:
                    return 0


# Check if the teams played as a champion or relegated.
league_table['relegated'] = league_table.apply(
    lambda x: relegation_championship_check(x) if x['round_type'] == 'Relegation Round' else None, axis=1)
print("champion_relegated_methods.py  :  league_table['relegated']  -  DONE")


league_table['champion'] = league_table.apply(
    lambda x: relegation_championship_check(x) if x['round_type'] == 'Championship Round' else None, axis=1)
print("champion_relegated_methods.py  :  league_table['champion']  -  DONE")

print()