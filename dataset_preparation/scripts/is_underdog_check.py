import pandas as pd

master_df = pd.read_csv('../dt_prep_tables/master_data_temp_test_21_05.csv')
# master_df = pd.read_csv('../dt_prep_tables/master_new_data_temp_1205.csv')  # Relevant for only new data


def get_position(df, team):
    """
    Returns the position of a team in the league table of a given data frame.
    :param df: dataFrame
    :param team: string. The team that need to get the position of.
    :return: int. Team's position.
    """
    if df[df['home_team'] == team].empty:  # Check if the team is on the away or home team.
        return df[df['away_team'] == team]['away_position'].values
    return df[df['home_team'] == team]['home_position'].values


def in_pre_season(df, team):
    # Returns True if a team played in the given data frame, which represents a previous season or seasons.
    res = team in df[['home_team', 'away_team']].values
    return res


def get_filtered_df(top_or_bottom, equal, pre_season):
    """
    Returns filtered df as the conditions said.
    :param top_or_bottom: string. Gets if the requested data is in the top or bottom playoff.
    :param equal: bool. Gets if the requested data needed to be in one or more previous seasons.
    :param pre_season: int. Get the season for the requested data.
    :return: dataFrame. Filtered df as the conditions said.
    """
    # Saves the round type and the last round of the round type according to the given points.
    round_type = 'Relegation Round' if top_or_bottom == 'bottom' else 'Championship Round'
    round_num = 7 if top_or_bottom == 'bottom' else 10
    # Filters the df accordingly.
    df = master_df[(master_df['round_type'] == round_type) & (master_df['round'] == round_num)]
    if equal:
        df = df[df['season'] == pre_season]  # Needed just the previous season.
    else:
        df = df[df['season'] < pre_season]  # Needed all previous seasons.
    return df


def get_max_season(df, team):
    # Return the last season that the given team was in.
    return max(df[(df['home_team'] == team) | (df['away_team'] == team)]['season'])


def is_underdog(df):
    # A :
    # For the first round of every season, besides the first season in the record, there is no telling which team is
    # the underdog team in the game because there are no positions at the beginning of the season.
    # Because of that, the underdog team is determined by the last round of the previous season's positions.

    # B :
    # For the rest of the records in the data set the determining of the underdog team is for the team that positioned
    # lower at the end of the previous round.

    # The table below represents all the options for determining the underdog team for the A options.
    # p.off = playoff ; lg = league ; h = home team ; a = away team
    # pst rltd = position related ; ssn rltd  = season related

    # | options    |  opt.1  |  opt.2  |  opt.3  |  opt.4  |  opt.5  |   opt.6  |  opt.7  |  opt.8  |   opt.9  |
    # |--------------------------------------------------------------------------------------------------------|
    # | top p.off  |  h & a  |    h    |    h    |    a    |    a    |          |         |         |          |
    # | bottom p.o |         |    a    |         |    h    |         |   h & a  |    h    |    a    |          |
    # | not in lg  |         |         |    a    |         |    h    |          |    a    |    h    |   h & a  |
    # |========================================================================================================|
    # | underdog  | pst rltd |    a    |    a    |    h    |    h    | pst rltd |    a    |    h    | pst rltd |

    # If both teams are in the same playoff or both were not in the league in the previous season the underdog is
    # based on the team's positions

    # opt.9 has 4 sub-options:
    # | options                         |  opt.9.1  |  opt.9.2  |  opt.9.3  |  opt.9.4  |
    # |---------------------------------------------------------------------------------|
    # | were sometime in the league     |   h & a   |     h     |           |     a     |
    # | were NOT sometime in the league |           |     a     |   h & a   |     h     |
    # |=================================================================================|
    # | underdog                        |  ssn rltd |     a     |   None *  |     h     |

    #   >> If both teams were some time in the league => determined by the latest season that each team was in.
    #       The team on the more previous season is the underdog (unless both teams were only on the same one previuos
    #       season the underdog is with the lower position in that season on the last round).
    #   >> If only one of the teams weren't in the league before => that team is the underdog.
    #   >> If both teams weren't in the league before => it can't be said which team is the underdog.(*)

    # Saves the previous season number, the home and away teams.
    pre_season = int(df['season']) - 1
    home_team = df['home_team']
    away_team = df['away_team']

    # Filters master_df to bottom and top playoff in the previous season.
    prev_season_bottom = get_filtered_df('bottom', True, pre_season)
    prev_season_top = get_filtered_df('top', True, pre_season)

    # Check if the record is in the first round of the Regular season (options B)
    if df['round_type'] == 'Regular Season' and df['round'] == 1:
        if df['season'] > 2012 or df['season'] > 2022:
            # Checks if both home and away teams are in the top playoff.
            if in_pre_season(prev_season_top, home_team) and in_pre_season(prev_season_top, away_team):
                # Gets the position of both teams.
                home_pos = get_position(prev_season_top, home_team)
                away_pos = get_position(prev_season_top, away_team)
                return home_team if home_pos > away_pos else away_team  # opt.1
            # Checks if both home and away teams are in the bottom playoff.
            elif in_pre_season(prev_season_bottom, home_team) and in_pre_season(prev_season_bottom, away_team):
                home_pos = get_position(prev_season_bottom, home_team)
                away_pos = get_position(prev_season_bottom, away_team)
                return home_team if home_pos > away_pos else away_team  # opt.6
            # Checks if both home and away teams weren't in the league in the previous season.
            elif not in_pre_season(prev_season_top, home_team) and not in_pre_season(prev_season_bottom,
                                                                                     home_team) and not in_pre_season(
                    prev_season_top, away_team) and not in_pre_season(prev_season_bottom, away_team):
                # Gets the previous seasons.
                prev_season_bottom = get_filtered_df('bottom', False, pre_season)
                prev_season_top = get_filtered_df('top', False, pre_season)
                prev_seasons = pd.concat([prev_season_bottom, prev_season_top])
                # Checks if both teams were in the league before the previous season.
                if in_pre_season(prev_seasons, home_team) and in_pre_season(prev_seasons, away_team):  # opt.9.1
                    if get_max_season(prev_seasons, home_team) > get_max_season(prev_seasons, away_team):
                        return home_team
                    elif get_max_season(prev_seasons, home_team) < get_max_season(prev_seasons, away_team):
                        return away_team
                    else:
                        home_pos = get_position(prev_seasons, home_team)
                        away_pos = get_position(prev_seasons, away_team)
                        return home_team if home_pos > away_pos else away_team
                    # return home_team if get_max_season(prev_seasons, home_team) > get_max_season(prev_seasons, away_team) else away_team
                # Checks if the home team was in the league before the previous season.
                elif in_pre_season(prev_seasons, home_team):  # opt.9.2
                    return away_team
                # Checks if the away team was in the league before the previous season.
                elif in_pre_season(prev_seasons, away_team):  # opt.9.4
                    return home_team
                # Both teams were'nt in the league before the previous season.
                else:  # opt.9.3
                    return None
            # Checks if the home team was in the top playoff.
            elif in_pre_season(prev_season_top, home_team):
                return away_team  # opts.2 & opt.3
            # Checks if the away team was in the top playoff.
            elif in_pre_season(prev_season_top, away_team):
                return home_team  # opts.4 & opt.5
            # Checks if the away team was in the bottom play off or not.
            else:
                return home_team if in_pre_season(prev_season_bottom, away_team) else away_team
        return None  # To all the records of the first round of the 2012 season.
    else:  # The record is not in the first round of the Regular season (options A)
        if df['home_pos_b4_game'] > df['away_pos_b4_game']:
            return home_team
        return away_team


def get_pre_pts(round_num, season, round_type, team):
    """
    Gets the points in the previous round.
    :param round_num: int. the current round number (the point to get is previous to this round)
    :param season: int.
    :param round_type: string. The round type name.
    :param team: string. who's point to get?
    :return: int. the points of the previous round of the team.
    """
    # Saves the round type and the last round of the round type according to the given points.
    round_num = round_num - 1
    # Filters the df accordingly.
    df = master_df[(master_df['round_type'] == round_type) & (master_df['round'] == round_num) & (master_df['season'] == season)]
    # Checks if the given team was away or home team in the previous round.
    if df[df['home_team'] == team].empty:  # The team wasn't the home team => it was in the away team.
        df = df[df['away_team'] == team]
        return int(df['away_league_pts'].values)
    else:  # The team was home team.
        df = df[df['home_team'] == team]
        return int(df['home_league_pts'].values)


def pts_b4_game_home(record):
    # Gets the points of the previous round pts of the home team.
    if record['round'] == 1:
        return 0
    else:
        return get_pre_pts(record['round'], record['season'], record['round_type'], record['home_team'])


def pts_b4_game_away(record):
    # Gets the points of the previous round pts of the away team.
    if record['round'] == 1:
        return 0
    else:
        return get_pre_pts(record['round'], record['season'], record['round_type'], record['away_team'])
