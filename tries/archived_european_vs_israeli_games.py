import pandas as pd
from datetime import datetime, timedelta

leagues_files = ["bundesliga.csv", "laliga.csv", "league_1.csv", "premier_league.csv", "serie_a.csv"]
path = "C:/Users/talik/OneDrive - ac.sce.ac.il/SCE/Final_Project/web scraping/14_03_2023/"
earopean_leagues = pd.DataFrame()

# Appending the data of the cvs files to a data frame
for file in leagues_files:
    location = path + str(file)
    df_temp = pd.read_csv(location)
    earopean_leagues = earopean_leagues.append(df_temp, ignore_index=True)

earopean_leagues = earopean_leagues[['League', 'matchDate', 'matchHour']]


def to_time(s):
    """
    Converting value type string to time.
    :param s: string. The string ro convert.
    :return: datetime.datetime object.
    """
    try:
        form = '%H:%M:%S'
        return datetime.strptime(s, form)
    except:  # If the string is empty, the function will sets it as 00:00:00 time.
        form = '%H:%M:%S'
        return datetime.strptime('00:00:00', form)


def to_date(s):
    """
    Converting value type string to date.
    :param s: string. The string ro convert.
    :return: datetime.datetime object.
    """
    try:  # 1st option of date form.
        form = '%d/%m/%Y'
        return datetime.strptime(s, form).date()
    except:  # 2nd oprion of date form.
        form = '%Y-%m-%d'
        return datetime.strptime(s, form).date()


def add_time_h(t, x):
    """
    Adds x minutes to a givven time.
    :param t: datetime.datime.
    :param x: int. The number of minuts needed to add.
    :return: datetime.datime. The new time.
    """
    return t + timedelta(hours=x)


earopean_leagues['matchHour'] = earopean_leagues.apply(lambda x: to_time(x['matchHour']), axis=1)
earopean_leagues['matchDate'] = earopean_leagues.apply(lambda x: to_date(x['matchDate']), axis=1)
earopean_leagues['matchHour'] = earopean_leagues.apply(lambda x: (x['matchHour'] + timedelta(hours=2))
                                                        if x['League'] == 'Premier League'
                                                        else (x['matchHour'] + timedelta(hours=1)), axis=1)


def date_filter(d):
    """
    Filters the df "earopean_leagues" to match a givven date.
    :param d: string. The string of the date that needed to be filtered by.
    :return: Dataframe. Filterd df "earopean_leagues".
    """
    return earopean_leagues[(earopean_leagues['matchDate'] == d)]


def add_time_m(t, x):
    """
    Adds x minutes to a given time.
    :param t: datetime.datime.
    :param x: int. The number of minuts needed to add.
    :return: datetime.datime. The new time.
    """
    return t + timedelta(minutes=x)


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


def meanwhile_european_game(record):
    """
    Check if the given game is meanwhile at least one of the europen games.
    :param record: series. A record from the df to do the check.
    :return: series. The record with additional column that saves the result of the check.
    """
    flag = False
    filterd_df = date_filter(record['date'])  # saves the relevant records by the date.

    kot_israeli = to_time(str(record['KOT']))  # kot of Ligat Ha'al game.
    end_israeli = (kot_israeli + timedelta(minutes=105))  # End of Ligat Ha'al game (Adding 105 minutes to the kot)
    # Taking only the kot column.
    european = pd.DataFrame(filterd_df['matchHour'])
    # Adding to the kot column 105 minutes to save the end time.
    european['end_hour'] = european.apply(add_time_m, args=(105,))

    # Loop over the games, to check if there is at leat one european game that colliding with the Israeli game.
    for i in european.index:
        european_i_s = european.at[i, 'matchHour']  # Eueopean game kot.
        european_i_f = european.at[i, 'end_hour']  # European game end time.

        if european_i_s > kot_israeli:  # Checks if the Israeli game started earlier.
            if kot_israeli < european_i_s < end_israeli:  # Checks if the European game started while the Israeli game.
    # Checks if the differance is bigger that 5 minutes (for the records where the kot is not in in 5 minute increments.
                diff = time_difference_duration(end_israeli, european_i_s)
                if diff > 5:  # The difference is larger than 5 minutes => the game was while a Israeli game.
                    flag = True
                    break  # It's enough that one European game ware while the Israeli game.
                else:
                    flag = False
            else:
                flag = False
        else:
            if european_i_s < kot_israeli < european_i_f:  # Checks if the Israeli game started while the European game.
                diff = time_difference_duration(european_i_f, kot_israeli)
                if diff > 5:
                    flag = True
                    break
                else:
                    flag = False
            else:
                flag = False

    if flag:  # Sets the appropriate value to the record.
        record['meanwhile_european_games'] = 1
    else:
        record['meanwhile_european_games'] = 0
    return record
