from datetime import datetime, timedelta
import pandas as pd


def while_world_cup(d):
    """
    Checks if the match took place while the world cups were on (in 2014 and 2018).
    :param d: date. Game's date that is checked if took place while one of the World Cups.
    """
    # List of the start dates of the World Cups of 2014 and 2018.
    world_cup_start = list(map(lambda x: datetime.strptime(x, '%d-%m-%Y').date(), ['12-06-2014', '14-06-2018']))
    # List of the end dates of the World Cups of 2014 and 2018.
    world_cup_end = list(map(lambda x: datetime.strptime(x, '%d-%m-%Y').date(), ['13-07-2014', '15-07-2018']))
    return sum(list(map(lambda x, y: 1 if x < d < y else 0, world_cup_start, world_cup_end)))


# List of the beginning of each kot category (i.e. 15-16)
start = list(map(lambda x: datetime.strptime(x, '%H:%M:%S').time(),
                 ['15:00:00', '16:00:00', '17:00:00', '18:00:00', '19:00:00', '20:00:00', '21:00:00', '22:00:00']))
# List of the end of each kot category (i.e. 15-16)
end = list(map(lambda x: datetime.strptime(x, '%H:%M:%S').time(),
               ['16:00:00', '17:00:00', '18:00:00', '19:00:00', '20:00:00', '21:00:00', '22:00:00', '23:00:00']))


def kot_determining(t):  # new master df
    """
    Determining the kot category (i.e. 15-16 for kot = 15:50, 19-20 for kot = 19:00)
    :param t: datetime. Kot of the game to categorize.
    :return: string. The category of the given kot.
    """
    t = t.time()  # Converting from datetim.datetime into datetime.time.
    # Checkes where 't' is between 'start' and 'end' items in each list.
    # For where the 't' is, the function saves the hour in the list 'temp'
    # and for all the other hours save 'None' in the list 'temp'.
    temp = list(map(lambda x, y: x.strftime('%H') if x <= t < y else None, start, end))
    res = [x for x in temp if x is not None].pop()  # Extracting the hour between all the None items.
    return str(res) + '_' + str(int(res) + 1)


def to_time(s):
    """
    Converts value type form string to datetime.
    :param s: string. The string to convert.
    :return: datetime.datetime object.
    """
    try:  # If the string form is X:XX AM/PM
        return datetime.strptime(s, "%I:%M %p")
    except:
        try:  # If the string form is X:XX Am/PM 1st leg:
            s_ = s[:7]
            return datetime.strptime(s_, "%I:%M %p")
        except:  # If the string form is XX:XX:XX
            form = '%H:%M:%S'
            return datetime.strptime(s, form)


def to_date(s):
    """
    Converts value type form string to datetime.date.
    :param s: string. The string to convert.
    :return: datetime.date object.
    """
    try:  # 1st option of date form.
        form = '%d/%m/%Y'
        return datetime.strptime(s, form).date()
    except:  # 2nd oprion of date form.
        form = '%Y-%m-%d'
        return datetime.strptime(s, form).date()


# Import tables and turn them into a dataframe
champion_league = pd.read_csv(r'data_tables/ChampinLeagueFinalFixed.csv')

leagues_files = ["bundesliga.csv", "laliga.csv", "league_1.csv", "premier_league.csv", "serie_a.csv"]
path = "C:/Users/talik/OneDrive - ac.sce.ac.il/SCE/Final_Project/web scraping/14_03_2023/"
earopean_leagues = pd.DataFrame()

# Appending the data of the cvs files to a data frame
for file in leagues_files:
    location = path + str(file)
    df_temp = pd.read_csv(location)
    earopean_leagues = earopean_leagues.append(df_temp, ignore_index=True)

earopean_leagues = earopean_leagues[['matchDate', 'matchHour']]  # Extracting onlty the relevant columns.
earopean_leagues.rename(columns={'matchDate': 'match_date', 'matchHour': 'kot'}, inplace=True)

# Converting the columns values into time type.
earopean_leagues['kot'] = earopean_leagues['kot'].apply(to_time)
champion_league['kot'] = champion_league['kot'].apply(to_time)
# Adjusting to Isreal's time.
earopean_leagues['kot'] = earopean_leagues['kot'].apply(lambda x: (x + timedelta(hours=1)))
champion_league['kot'] = champion_league['kot'].apply(lambda x: x + timedelta(hours=1))
# Adding 105 minutes to the kot to determine game end time.
earopean_leagues['end'] = earopean_leagues['kot'].apply(lambda x: x + timedelta(minutes=105))
champion_league['end'] = champion_league['kot'].apply(lambda x: x + timedelta(minutes=105))
# Converting the columns values into date type.
earopean_leagues['match_date'] = earopean_leagues['match_date'].apply(to_date)
champion_league['match_date'] = champion_league['match_date'].apply(to_date)


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
df_dct = {'champion_league': champion_league, 'earopean_leagues': earopean_leagues}


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