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
        pos_6 = int(league_table[(league_table['game'] == record['game']) &
                                 (league_table['season'] == record['season']) &
                                 (league_table['round'] == (record['round']) - 1) &
                                 (league_table['team_pos'] == 6)]['pts'])

        # team_pos is the number of pts that the team had on the previous round of the given record's
        # (the team's places are 7th or 8th).
        team_pos = int(league_table[(league_table['game'] == record['game']) &
                                    (league_table['season'] == record['season']) &
                                    (league_table['round'] == (record['round']) - 1) &
                                    (league_table['clubs_name'] == record['clubs_name'])]['pts'])
        diff = (pos_6 - team_pos)
        return diff

    else:  # code = 10 => top playoff (championship)
        # For top playoff the importent pts difference is between 1st and 2nd places.
        # pos_1 is the number of pts that the team had on the previous round of the given record's that was in 1st place
        pos_1 = int(league_table[(league_table['game'] == record['game']) &
                                 (league_table['season'] == record['season']) &
                                 (league_table['round'] == (record['round']) - 1) &
                                 (league_table['team_pos'] == 1) &
                                 (league_table['clubs_name'] == record['clubs_name'])]['pts'])

        # pos_2 is the number of pts that the 2nd place team had on the previous round of the given record's.
        pos_2 = int(league_table[(league_table['game'] == record['game']) &
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
    if record['game'] == 'Relegation Round':  # Checks if the game is in the relegation round.
        if record['round'] > 4:  # The relegation check is relevant from the 5th round.
            if record['team_pos'] > 6:  # The relegation check is relevant for the 7th and 8th places.
                max_pts = 3 * (7 - (record['round'] - 1))  # Max pts that the team can win until the end of the season.
                if pts_difference(7, record) > max_pts:
                    # If the pts difference between 6th place and the given team's place is bigger than the max pts that
                    # the team can win => the given team is for sure relegated.
                    return 1
                else:  # If NOT => It can't be said for sure if the given team is relegated.
                    return 0

    elif record['game'] == 'Championship Round':  # Checks if the game is in the championship round.
        if record['round'] > 5:  # The championship check is relevant from the 7th round.
            if record['team_pos'] == 1:  # The championship check is relevant only for the first place in the round.
                a = league_table[(league_table['game'] == record['game']) &
                                 (league_table['season'] == record['season']) &
                                 (league_table['round'] == (record['round']) - 1) &
                                 (league_table['team_pos'] == 1) &
                                 (league_table['clubs_name'] == record['clubs_name'])]
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