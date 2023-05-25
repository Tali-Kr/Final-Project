def time_difference_duration(t_1, t_2):
    """
    Gets the difference between 2 times in minutes.
    :param t_1: datetime.datime. 1st time from needed to subtract the other time.
    :param t_2: datetime.datime. 2nd time the needed to subtracted.
    :return: int. The difference in minutes.
    """
    diff = t_1 - t_2
    duration_in_s = diff.total_seconds()  # Converting the difference into minutes.
    return divmod(duration_in_s, 60)[0]


# Creating dictionary to store the names of the dataframes as 'keys' and the dataframes itself as the 'values'.
df_dct = {'champion_league': champion_league, 'european_leagues': earopean_leagues}


def while_champion_european_leagues(t, d, code):
    """
    Checks if given date and time of an Israeli game were played while the 'Champion Leauge' or on of the european leauges based on given code.
    :param t: kot of the Israeli game.
    :param d: The date of the Israeli game.
    :param code: string. The name of the league to check.
    :return: If the given game was played while at least one European  game or the 'Champion League' game => 1
                    otherwise => 0
    """
    df = df_dct.get(code)  # df is champion_league or european_leagues
    # Filter the matching to the given code dataframe by the given date and time.
    temp = df.loc[(df['match_date'] == d) &
                  (((df['kot'] < t) & (t < df['end'])) |  # For Israeli's games that started while the European game.
                   ((t < df['kot']) & (df['kot'] < t + timedelta(
                       minutes=105))))]  # For European games that started while the Israeli game.
    if temp.empty:  # If there are no games that meet these terms the Israeli game wasnt played while European game
        return 0
    else:  # If there are games that meet these terms

        # The difference between the start of the later game and the end of the earlier game.
        temp['diff'] = temp.apply(lambda x: time_difference_duration(x['end'], t) if x['end'] > t
        else time_difference_duration(t + timedelta(minutes=105), x['kot']), axis=1)

        # Checks whether the difference is bigger than 5 minutes.
        # for the records where the kot is not in in 5 minute increments. (e.i. : kot = 16:31 instead of 16:30)
        temp = temp[temp['diff'] > 5]

        if temp.empty:
            return 0
        else:
            return 1