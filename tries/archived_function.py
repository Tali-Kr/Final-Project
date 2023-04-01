from champion_relegated_methods import *
from datetime import datetime, date


def convert_timestamp_to_date(df, index):
    '''
    Converting timestamp of the game date to date in "YYYY-MM-DD" form.
    :param df:
    :param index: int. Range of the loop
    '''
    form = '%d-%m-%Y'
    for i in range(index):
        df.at[i, 'date'] = datetime.fromtimestamp(df.at[i, 'timestamp']).strftime(form)
        df.at[i, 'date'] = datetime.strptime(df.at[i, 'date'], form).date()

def get_pts_by_position(df, season, round_num, pos, pos_num, league_pts, game):
    """
    Gets the number of points of a given team's position, season and round in the league table.
    :param df: dataframe. The target dataframe's name.
    :param season: int. Year of the season.
    :param round_num: int. Number of round.
    :param pos: string. The name of the wanted coulmn (home/away_pos).
    :param pos_num: int. The wanted team's position to find the points number.
    :param league_pts: string. The target column's name.
    :param game: string. The round's name.
    :return: int. The number of points by position.
    """
    # filters the dataframe by the relavent values.
    value = df[(df['round'] == (game + str(round_num))) &
               (df['season'] == season) & (df[pos] == pos_num)]
    return int(value[league_pts])


def get_record_position(df, season, round_num, pos, pos_num, game):
    """
    Gets a record of the given season, round and team's position.
    :param df: dataframe. The target dataframe's name.
    :param season: int. Year of the season.
    :param round_num: int. Number of round.
    :param pos: string. The name of the wanted coulmn (home/away_pos).
    :param pos_num: int. The wanted team's position to find the points number.
    :param game: string. The round's name.
    :return: dataframe. The wanted record.
    """
    # filters the dataframe by the relavent values.
    return df[(df['round'] == (game + str(round_num))) &
              (df['season'] == season) & (df[pos] == pos_num)]


def get_record_by_team_name(df, season, round_num, h_a_team_name, team_name, game):
    """
    Gets a record of the given season, round and team's name.
    :param df: dataframe. The target dataframe's name.
    :param season: int. Year of the season.
    :param round_num: int. Number of round.
    :param h_a_team_name: string. Column's name of the wanted team (away/home)
    :param team_name: string. The name of the wanted team.
    :param game: string. The round's name.
    :return: dataframe. The wanted record.
    """
    return df[df['season'] == season & df['round'] == (game + str(round_num)) & df[h_a_team_name] == team_name]


def get_record_index_by_team_key(df, team_key):
    try:
        home = df[df['key_home'] == team_key]
        away = df[df['key_away'] == team_key]
        if is_in(team_key, df['key_home']):
            return home.index[0]
        else:
            return away.index[0]
    except:
        league = df[df['key'] == team_key]
        return league.index[0]


def get_round_name(round_name):
    """
    Gets round's name for a given round (Separates the name of the round from its number).
    :param round_name: string. given round.
    :return: string. the rounds name
    """
    if 'Championship' in round_name:
        return 'Championship Round - '
    elif 'Relegation' in round_name:
        return 'Relegation Round - '
    else:
        return 'Regular Season - '


def differnce_pts_st_nd(df, season, round_num, game):
    """
    Finds the difference between 2 first places' points in a certain season and round.
    :param df: dataframe. The target dataframe's name.
    :param season: int. Year of the season.
    :param round_num: int. Number of round.
    :param game: string. The round's name.
    :return: int. The difference between the 2 first places in the season and round.
    """
    st_pos = 0  # first position' points.
    nd_pos = 0  # second position' points.
    for i in range(1, 3):  # only relevent for 2 first positions, 'i' goes from 1 to 2.
        # checks if the the first position of the given season and round is the home's team.
        if get_record_position(df, season, round_num, 'away_pos', i, game).empty:  # if the record is empty ->
            # the position is the home's team.
            # distinguish between first and second position.
            if i == 1:
                # gets the points of first and second positions to claculate the difference.
                st_pos = get_pts_by_position(df, season, round_num, 'home_pos', i, 'home_league_pts', game)
            else:
                nd_pos = get_pts_by_position(df, season, round_num, 'home_pos', i, 'home_league_pts', game)
        else:  # the record is not empty -> the position is the away's team.
            if i == 1:
                st_pos = get_pts_by_position(df, season, round_num, 'away_pos', i, 'away_league_pts', game)
            else:
                nd_pos = get_pts_by_position(df, season, round_num, 'away_pos', i, 'away_league_pts', game)
    return st_pos - nd_pos


def promotion_check(df, season, round_num, pos_num, game):
    """
    Checks whether a team has already been promoted (won the season) in rounds before the last one
    :param df: dataframe. The target dataframe's name.
    :param season: int. Year of the season.
    :param round_num: int. Number of round.
    :param pos_num: int. The wanted team's position to find if promoted.
    :param game: string. The round's name.
    :return: 1 - if in season and round the team already been promoted (won the season).
             0 - if not.
    """
    # the check is only relevant to the first position, since it can only be determined with certainty for the first
    # position.
    if pos_num == 1:
        # the team in the first position in the league table will win for sure if the point differnece between the
        # first and secon position is larger than maximum point that the second position's team can get in the
        # remained rounds. Where the maximum points are calculated as :
        # number of points that a winning team gets * (total number of rounds in the season - current season) = 3 * (10 - round_n)
        if differnce_pts_st_nd(df, season, round_num, game) > (
                3 * (10 - round_num)):  # if the first positioned team promoted in the round
            return 1
        else:  # if not
            return 0
    else:
        return 0


def differnce_pts_2_last(df, season, round_num, i, game):
    if get_record_position(df, season, round_num, 'away_pos', 6, game).empty:
        pos_6 = get_pts_by_position(df, season, round_num, 'home_pos', 6, 'home_league_pts', game)
    else:
        pos_6 = get_pts_by_position(df, season, round_num, 'away_pos', 6, 'away_league_pts', game)

    if get_record_position(df, season, round_num, 'away_pos', i, game).empty:
        pos_i = get_pts_by_position(df, season, round_num, 'home_pos', i, 'home_league_pts', game)
    else:
        pos_i = get_pts_by_position(df, season, round_num, 'away_pos', i, 'away_league_pts', game)

    return pos_6 - pos_i


def relegation_check(df, season, round_num, pos, game):
    if pos > 6:
        if differnce_pts_2_last(df, season, round_num, pos, game) > (3 * (7 - (round_num - 1))):
            return 1
        else:
            return 0

    else:
        return 0


def promotion_check_acctivater(df, i):
    if len(df.at[i, 'round']) == 22 or len(df.at[i, 'round']) == 18 or len(df.at[i, 'round']) == 20:
        d = 1
    else:
        d = 2
    game = get_round_name(df.at[i, 'round'])

    if game == 'Championship Round - ':
        if int(df.at[i, 'round'][-d:]) >= 6:
            df.at[i, 'home_promoted'] = promotion_check(df, df.at[i, 'season'], int(df.at[i, 'round'][-d:]),
                                                        df.at[i, 'home_pos'], game)

            df.at[i, 'away_promoted'] = promotion_check(df, df.at[i, 'season'], int(df.at[i, 'round'][-d:]),
                                                        df.at[i, 'away_pos'], game)
        else:
            df.at[i, 'home_promoted'] = 0
            df.at[i, 'away_promoted'] = 0

    elif game == 'Relegation Round - ':
        if int(df.at[i, 'round'][-d:]) >= 4:
            df.at[i, 'home_relegated'] = relegation_check(df, df.at[i, 'season'], int(df.at[i, 'round'][-d:]),
                                                          df.at[i, 'home_pos'], game)

            df.at[i, 'away_relegated'] = relegation_check(df, df.at[i, 'season'], int(df.at[i, 'round'][-d:]),
                                                          df.at[i, 'away_pos'], game)
        else:
            df.at[i, 'home_relegated'] = 0
            df.at[i, 'away_relegated'] = 0
    print(i)


def get_home_or_away(df, season, round_num, game, team_name):
    """
    Gets the name of the column of the given team in a record.
    :param df: dataframe. The target dataframe's name.
    :param season: int. Year of the season.
    :param round_num: int. Number of round.
    :param game: string. The round's name.
    :param team_name: string. The given team's name.
    :return: string. The name of the coulmn of where the team is in the given season and round.
    """
    if df[df['season'] == season & df['round'] == (game + str(round_num)) & df['awayTeamName'] == team_name].empty:
        return 'homeTeamName'
    else:
        return 'awayTeamName'


def get_team_name(df, season, round_num, game):
    if df[df['season'] == season & df['round'] == (game + str(round_num))]:
        return 0
    return 8


def get_record_by_team_name_test(df, season, round_num, team_name, game):
    c = get_home_or_away(df, season, round_num, game, team_name)

    return df[(df['season'] == season) & (df['round'] == (game + str(round_num))) & (df[c] == team_name)]


def test_relegation_get_team_pts_by_key(df, key):
    return int(df[df['key'] == key]['pts'])


#################################

def differnce_pts_2_last_test(df, season, round_num, i, game):
    pos_6 = get_pts_by_position_test(df, season, round_num, 6, game)
    pos_i = get_pts_by_position_test(df, season, round_num, i, game)
    return pos_6 - pos_i


def get_pts_by_position_test(df, season, round_num, pos_num, game):
    """
    Gets the number of points of a given team's position, season and round in the league table.
    :param df: dataframe. The target dataframe's name.
    :param season: int. Year of the season.
    :param round_num: int. Number of round.
    :param pos_num: int. The wanted team's position to find the points number.
    :param game: string. The round's name.
    :return: int. The number of points by position.
    """
    # filters the dataframe by the relavent values.
    value = df[(df['round'] == round_num) &
               (df['season'] == season) & (df['team_pos'] == pos_num) & (df['game'] == game)]
    return int(value['pts'])


def relegation_check_test(df, season, round_num, pos, game):
    if pos > 6:
        if differnce_pts_2_last_test(df, season, round_num - 1, pos, game) > (3 * (7 - (round_num - 1))):
            return 1
        else:
            return 0
    else:
        return 0


league_tables = pd.read_csv('league_tables_test.csv')

for i in range(429, 431):
    if league_tables.at[i, 'game'] == 'Relegation Round':
        league_tables.at[i, 'played_the_game_already_relegated'] = \
            relegation_check_test(league_tables, league_tables.at[i, 'season'], league_tables.at[i, 'round'],
                                  league_tables.at[i, 'team_pos'], league_tables.at[i, 'game'])
    print(i)

    if relegation_check_test(league_tables, league_tables.at[i, 'season'], league_tables.at[i, 'round'],
                             league_tables.at[i, 'team_pos'], league_tables.at[i, 'game']) == 1:
        print(str(i) + "The game in round _ " + str(league_tables.at[i, 'round']) + "_  in season :  _" +
              str(league_tables.at[i, 'season']) + "_ for team : _" +
              str(league_tables.at[i, 'clubs_name']) +
              " _ was played when the team was relegated from the privos round.")
    else:
        print(str(i) + "The game in round _ " + str(league_tables.at[i, 'round']) + "_  in season :  _" +
              str(league_tables.at[i, 'season']) + "_ for team : _" +
              str(league_tables.at[i, 'clubs_name']) +
              "  _ was played when the team was !! NOT !! relegated from the privos ""round.")

print()

############ master_data_set ###############
print()
# def is_derby(home_clubs, away_clubs):
#     """
#     Checks if a game is a derby one. Meaning the function checks if the home team and the away team are from the same city.
#     :param home_clubs: List. A list of home cities
#     :param away_clubs: List. A list of away cities
#     :return: 1 => if the game is a derby game.
#              0 => if the game is NOT a derby game.
#     """
#     home_cities = get_cities(home_clubs)
#     away_cities = get_cities(away_clubs)
#     for i in range(len(home_clubs)):
#         # There is only one groupe that didnt have the name of the city in the club' name.
#         if home_cities[i] == 'Katamon' and away_cities[i] == 'Jerusalem' or home_cities[i] == 'Jerusalem' \
#                 and away_cities[i] == 'Katamon':  # Cheks if the game is a "Jerusalem derby".
#             master_df.at[i, 'derby'] = 1
#         elif home_cities[i] == away_cities[i]:  # Checks the rest of the cities derbies.
#             master_df.at[i, 'derby'] = 1
#         else:  # The game isn't a derby game.
#             master_df.at[i, 'derby'] = 0


# is_derby(master_df.homeTeamName, master_df.awayTeamName)
#
# def get_cities__1(club_name):
#     """
#     Extracts the city's name from the club's name.
#     All of the clubs have their home city in the club's name, most of the time the name appears on the second word in the name.
#     :param club_name: string. A club's name needed to extract city name.
#     :return: string. A city's name.
#     """
#     # Preforming function split_club_name on the club's names.
#     # splited_club_name = split_club_name(club_name)
#     # if len(splited_club_name) == 1:  # Addressing the option that a club may have only a one-word name.
#     #     return str(splited_club_name)
#     # else:
#     #     return str(splited_club_name[1])
#
# def get_cities(club_name):
#     """
#     Extracts the city's name from the club's name.
#     All of the clubs have their home city in the club's name, most of the time the name appears on the second word in the name.
#     :param club_name: Series. A column from the dataframe that needs to extract cities names.
#     :return: List. A list of the cities.
#     """
#     club_name = list(map(split_club_name,
#                          club_name))  # Preforming function split_club_name on all the club' names in the given series.
#     temp_list = []
#     for club in club_name:  # Loop over the objects in the list on clubs.
#         if len(club) == 1:  # Addressing the option that a club may have only a one-word name.
#             temp_list.append(str(club))
#         else:
#             temp_list.append(str(club[1]))
#     return temp_list
#
#
#
# def is_derby(home_club, away_club):
#     """
#     Checks if a game is a derby one. Meaning the function checks if the home team and the away team are from the same city.
#     :param home_clubs: List. A list of home cities
#     :param away_clubs: List. A list of away cities
#     :return: 1 => if the game is a derby game.
#              0 => if the game is NOT a derby game.
#     """
#     home_city = get_cities(home_club)
#     away_city = get_cities(away_club)
#     # There is only one groupe that didnt have the name of the city in the club' name.
#     # Cheks if the game is a "Jerusalem derby".
#     if (home_city == 'Katamon' and away_city == 'Jerusalem') or (home_city == 'Jerusalem' and away_city == 'Katamon'):
#         return 1
#     elif home_city == away_city:  # Checks the rest of the city's derbies.
#         return 1
#     else:  # The game isn't a derby game.
#         return 0

print()

########### champion relegation mathods #############
print()
# def pts_difference(code, df, game, season, round_num, team):
#     if code == 7:  # code = 7 => bottom playoff (relegation)
#         pos_6 = df[
#             (df['game'] == game) & (df['season'] == season) & (df['round'] == round_num - 1) & (df['team_pos'] == 6)]
#         pos_6 = int(pos_6['pts'])
#         team_pos = df[
#             (df['game'] == game) & (df['season'] == season) & (df['round'] == round_num - 1) & (
#                     df['clubs_name'] == team)]
#         team_pos = int(team_pos['pts'])
#         diff = (pos_6 - team_pos)
#         return diff
#     else:  # code = 10 => top playoff (championship)
#         pos_1 = df[
#             (df['game'] == game) & (df['season'] == season) & (df['round'] == round_num - 1) & (df['team_pos'] == 1) & (
#                     df['clubs_name'] == team)]
#         pos_1 = int(pos_1['pts'])
#         pos_2 = df[
#             (df['game'] == game) & (df['season'] == season) & (df['round'] == round_num - 1) & (df['team_pos'] == 2)]
#         pos_2 = int(pos_2['pts'])
#         diff = (pos_1 - pos_2)
#         return diff
#
# def relegation_championship_check(record, df):
#     if record['game'] == 'Relegation Round':
#         if record['round'] > 4:
#             if record['team_pos'] > 6:
#                 max_pts = 3 * (7 - (record['round'] - 1))
#                 if pts_difference(7, df, record['game'], record['season'], record['round'],
#                                   record['clubs_name']) > max_pts:
#                     return 1
#                 else:
#                     return 0
#
#     elif record['game'] == 'Championship Round':
#         if record['round'] > 6:
#             if record['team_pos'] == 1:
#                 max_pts = 3 * (10 - (record['round'] - 1))
#                 if pts_difference(10, df, record['game'], record['season'], record['round'],
#                                   record['clubs_name']) > max_pts:
#                     return 1
#                 else:
#                     return 0
print()

########## world_cup methods - old version ##########
print()
# def convert_timestamp_to_date(tms):
#     '''
#     Converting timestamp of the game date to date in "YYYY-MM-DD" form.
#     :param df:
#     :param index: int. Range of the loop
#     '''
#     form = '%d-%m-%Y'
#     res = datetime.fromtimestamp(tms).strftime(form)
#     d: date =  datetime.strptime(res,form).date()
#     return d

# def convert_timestamps_to_time(df, index):
#     '''
#     Converting timestamp of the game date to time in "HH:MM:SS" form.
#     :param df:
#     :param index: int. Range of the loop
#     '''
    # timezone = pytz.timezone('Israel')
    # form = '%H:%M:%S'
    # for i in range(index):
    #     df.at[i, 'KOT'] = timezone.localize(datetime.fromtimestamp(df.at[i, 'timestamp'])).strftime(form)
    #     df.at[i, 'KOT'] = datetime.strptime(df.at[i, 'KOT'], form).time()


# def convert_timestamps_to_time(tms):
#     '''
#     Converting timestamp of the game date to time in "HH:MM:SS" form.
#     :param df:
#     :param index: int. Range of the loop
#     '''
#     timezone = pytz.timezone('Israel')
#     form = '%H:%M:%S'
#     res = timezone.localize(datetime.fromtimestamp(tms)).strftime(form)
#     return (datetime.strptime(res,form).time()


#  def timestamps_to_time(tms):  # no need in the new version
#     """
#     Converting timestamp of the game to time in "HH:MM:SS" form.
#     :param tms: timestamp that needed to convert to time.
#     :return: datetime.time. The kot of the game.
#     """
#     # timezone = pytz.timezone('Israel')
#     # form = '%H:%M:%S'
#     # res = timezone.localize(datetime.fromtimestamp(int(tms))).strftime(form)
#     # return datetime.strptime(res, form).time()
#
# def timestamp_to_date(tms):  # no need in the new version
#     """
#     Converting timestamp of the game to date in "dd-mm-YYYY" form.
#     :param tms: timestamp that needed to convert to date.
#     :return: datetime.date. The date of the game.
#     """
#     form = '%d-%m-%Y'
#     res = datetime.fromtimestamp(int(tms)).strftime(form)
#     return datetime.strptime(res, form).date()
#
# def determining_KOT(t, start, end):  # no need in the new version
#     """
#     Checking the KOT group.
#     :param start: time. Lower bound for the tested category.
#     :param end: time. Upper bound for the tested category.
#     :param t: time. The KOT of the tested match.
#     :return: if the KOT of the tested match fit in between the bounds => the tested catecory is matching - returning 1
#                                                                                          if not matching - returning 0.
#     """
#     if start <= t < end:
#         return 1
#     else:
#         return 0

print()
############ champion_relegated_methods #############
print()
def find_row(df_target, key_t, df_sorce, key_s, i):
    """
    Finds the index row number in the target df that is matchin to the key in the sorce df.
    :param df_target: dataframe. The df that we want to find the index row number.
    :param key_t: string. The name of the coulmn of the key match for thw sorce key.
    :param df_sorce: dataframe. The df that is the sorce of the key that we are looking for.
    :param key_s: string. The name of the coulmn of the key match for thw sorce key.
    :param i: int. The index of the row for a serch in the sorce df.
    :return: The index row number of the target df that contains the key of the sorce df.
    """
    value = df_sorce.at[i, key_s]
    row_number = df_target.loc[df_target[key_t] == value]
    return row_number.index[0]


def is_in(value, column):
    """
    Checks if a value is in a given coulmn.
    :param value: any (string or int). The value that we are looking for.
    :param column: series. The column that we want to check if the value in it.
    :return: True - if the value in the column   /   False - if the value is NOT in the column.
    """
    flag = False
    if value in column.values:
        flag = True
    return flag
