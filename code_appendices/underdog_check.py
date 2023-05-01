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
    # Saves the previous season number, the home and away teams.
    pre_season = int(df['season']) - 1
    home_team = df['home_team']
    away_team = df['away_team']

    # Filters master_df to bottom and top playoff in the previous season.
    prev_season_bottom = get_filtered_df('bottom', True, pre_season)
    prev_season_top = get_filtered_df('top', True, pre_season)

    # Check if the record is in the first round of the Regular season (options B)
    if df['round_type'] == 'Regular Season' and df['round'] == 1:
        if df['season'] > 2012:
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
            elif not in_pre_season(prev_season_top, home_team) and not in_pre_season(prev_season_bottom, home_team) and not in_pre_season(prev_season_top, away_team) and not in_pre_season(prev_season_bottom, away_team):
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