def previous_round_pos(record):
    """
    Getting team's position on the previous round.
    :param record: series.
    :return: int. the position on the previous round.
    """
    pos = None  # Reset of the variable.
    # All the rounds but the first of each round type (regular season, relegation or championship round).
    if record['round'] != 1:
        # Fillters out the relevant season, round type, team's name in the previous round, and gets the position.
        pos = league_table[(league_table['game'] == record['game']) &
                           (league_table['season'] == record['season']) &
                           (league_table['round'] == (record['round']) - 1) &
                           (league_table['clubs_name'] == (record['clubs_name']))]['team_pos']
        return int(pos.values)
    else:  # For the first round of each round type (regular season, relegation or championship round).
        # The check is irrelevant for the first round in the regular season.
        if record['game'] == 'Relegation Round' or record['game'] == 'Championship Round':
            pos = league_table[(league_table['game'] == 'Regular Season') &
                               (league_table['season'] == record['season']) &
                               (league_table['round'] == 26) &
                               (league_table['clubs_name'] == (record['clubs_name']))]['team_pos']
            return int(pos.values)
    return pos
