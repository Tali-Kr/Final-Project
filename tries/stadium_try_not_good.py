from datetime import timedelta

import numpy as np
import pandas as pd
from thefuzz import fuzz
from champion_relegated_methods import *
from archived_european_vs_israeli_games import to_date, to_time

from difflib import SequenceMatcher

# Importing relevent csv files to dataframes.
master = pd.read_csv('master_data.csv')
stadiums = pd.read_csv(r'C:\Users\talik\OneDrive - ac.sce.ac.il\SCE\Final_Project\web scraping\Stadiums_In_Israel.csv')
ligat_haal = pd.read_csv(
    r'C:\Users\talik\OneDrive - ac.sce.ac.il\SCE\Final_Project\web scraping\14_03_2023\ligat_haal.csv')


def to_time(t):
    """
    Converts given datetime.datetime parameter into datetime.time type.
    :param t: datetime.datetime. Parameter to convert.
    :return: datetime.time. Converted parameter.
    """
    return t.time()

def to_date(d):
    return d.date()

def add_1_hour(x):
    """
    Adds 1 hour to the givnen datetime.datetime.
    :param x: datetime.datetime. The time to add 1 hour to.
    :return: datetime.datetime. The added time.
    """
    return x + timedelta(hours=1)


# converting the column into datetime.datetime
ligat_haal['match_date'] = ligat_haal['match_date'].apply(to_date)
# converting the column into datetime.datetime
ligat_haal['kot'] = ligat_haal['kot'].apply(to_time)
# adding 1 hour to the 'kot' to match the Israeli time and the master df.
ligat_haal['kot'] = ligat_haal['kot'].apply(add_1_hour)
# converting the column into datetime.time
ligat_haal['kot'] = ligat_haal['kot'].apply(to_time)

print(0)

# flag_date = False
#
# # # uniq_dates = list(map(to_date,ligat_haal['match_date'].unique()))
# uniq_dates = list(ligat_haal['match_date'].unique())
#
# print(0)
#
# for d in uniq_dates:
#     flag = False
#     a = ligat_haal[(ligat_haal['match_date'] == d)]['stadium']
#     for s in range(len(a)-1):
#         if a.iloc[s] == a.iloc[s+1]:
#             flag = True
#             break
#     print(str(d) + "  :  " + str(flag))



def spilt_stadium_name(name):
    """
    Extracting the stadiums name alone, without the word 'Stadium' and without the name of the city in () if there is.
    :param name: string. The name of stadiums as apears in the dataframe.
    :return: string. The name of the stadiums.
    """
    return name.split(' (')[0].rsplit(' ', 1)[0]


def cut_stadium(stadium):
    """
    Extracting the stadiums name alone, without the word 'Stadium'.
    :param stadium: string. The name of stadiums as apears in the dataframe.
    :return: string. The name of the stadiums.
    """
    return stadium.rsplit(' ', 1)[0]


master['stadium_temp'] = master['stadium'].apply(spilt_stadium_name)  # Getting only the stadiums names.
stadiums['Stadium'] = stadiums['Stadium'].apply(cut_stadium)  # Getting only the stadiums names.

rows = [0, 1, 2, 3, 4, 5, 6, 7, 10, 13, 14, 15, 16, 17, 26, 28]  # The indexes of the relavent stadiums in Israel.
new = stadiums[stadiums.index.isin(rows)]  # Adding to a new df the only stadiums that relavent.
new = new.drop('#', axis='columns')  # Dropping unessery column.

# converting the datafram 'new' to a dictoanery.
# where: the keys of the dictionary are the stadium names
#        the values of the dictonary are the rest of the columns in the df.
stadium_dic = new.set_index('Stadium').T.to_dict('list')


# new.to_csv("stadium_only_in_master_df.csv")
def name_check(stadium):
    """
    Gets stadium name and returns the name that appears in stadium_dic.
    :param stadium: string. The givven name.
    :return: string. The name that appears in stadium_dic.
    """
    name = stadium
    # Checks what is the name that is given and changing it according to the conditions below:
    if stadium == 'Teddi Malcha' or stadium == 'Teddy-Kollek-Stadion':
        name = 'Teddy'
    elif stadium == 'Yaakov Turner Toto' or stadium == 'Toto Turner' or stadium == 'Toto Jacob Turner':
        name = 'Turner'
    elif stadium == 'Winner':
        name = 'Netanya'
    elif stadium == 'National Stadium Ramat':
        name = 'Ramat Gan'
    elif stadium == 'Akko Municipal':
        name = 'Acre Municipal'
    elif stadium == 'Sala':
        name = 'Sela'

    return name


ligat_haal['stadium'] = ligat_haal['stadium'].apply(cut_stadium)  # Getting only the stadiums names.
ligat_haal['stadium'] = ligat_haal['stadium'].apply(name_check)  # Changing the names of the stadiums as appears in stadium_dic.

print(0)


def attendance(record):
    """
    Gets a record of a df and returns the record with additional column of attandance.
    :param record: series.
    :return: series.
    """
    # saving the "key" paramter to idanify the same game in master df in the ligat_haal df.
    # The "key" is game's date, kot and sradium. There are no 2 games that answers this key.
    d = to_date(record['date'])  # saves record's date into a parameter.
    # t = convert_string_to_time(record['KOT']).time()  # saves record's time into a parameter.
    stadium = record['stadium_temp']  # saves record's stadium into a parameter.
    try:
        # Extacting the "attandance" of the game that answers to the 'key' and sets it's value into the record.
        record['attendance'] = ligat_haal[(ligat_haal['match_date'] == d) &
                                          (ligat_haal['stadium'] == stadium)]['attendance'].values[0]
        print(record.values[0])
        return record
    except:
        # t = convert_string_to_time(record['KOT'])
        # t_min = (t - timedelta(minutes=15)).time()
        # t_max = (t + timedelta(minutes=15)).time()
        record['attendance'] = ligat_haal[(ligat_haal['match_date'] == d) &
                                          (ligat_haal['stadium'] == stadium)]['attendance'].values[0]
        print(record.values[0])
        return record

### the best option i think.
def similar_option_3(record):
    """
    Getting a record from the dataframe, checks for the right stadiums key and adding 3 coulmns of new values of:
                                                                                                1. stadiums capacity
                                                                                                2. stadiums cooradinates
                                                                                                3. stadiums age
    :param record: sereies. Record from the dataframe
    :return: series. The new recorde with the new values.
    """
    if is_in(record['stadium_temp'], new):  # for all the stadiums that have the same name in 'master' as in 'new'
        # sets the capacity of the stadium from the dictionary into the record.
        record['stadium_capacity'] = stadium_dic.get(record['stadium_temp'])[0]
        # sets the cooradinates of the stadium from the dictionary into the record.
        record['stadium_cooradinates'] = stadium_dic.get(record['stadium_temp'])[2]
        # sets the age of the stadium from the dictionary into the record.
        record['stadium_age'] = 2023 - int(stadium_dic.get(record['stadium_temp'])[3])

    else:  # for all the stadiums that don't have the same name in 'master' as in 'new'
        if record['stadium_temp'] == 'Teddi Malcha':
            name = 'Teddy'
        elif record['stadium_temp'] == 'Yaakov Turner Toto' or record['stadium_temp'] == 'Toto Turner':
            name = 'Turner'
        elif record['stadium_temp'] == 'Winner':
            name = 'Netanya'
        elif record['stadium_temp'] == 'National Stadium Ramat':
            name = 'Ramat Gan'
        else:
            name = 'Kiryat Shmona Municipal'

        record['stadium_temp'] = name
        # sets the capacity of the stadium from the dictionary into the record.
        record['stadium_capacity'] = stadium_dic.get(name)[0]
        # sets the cooradinates of the stadium from the dictionary into the record.
        record['stadium_cooradinates'] = stadium_dic.get(name)[2]
        # sets the age of the stadium from the dictionary into the record.
        record['stadium_age'] = 2023 - int(stadium_dic.get(name)[3])

    return record








master_try = master.apply(similar_option_3, axis=1) # and master.apply(attendance, axis=1)
# master_try.to_csv("master_try_stadium.csv")


# d = convert_string_to_date(master_try.at[0,'date'])  # saves record's date into a parameter.
# t = convert_string_to_time(master_try.at[0,'KOT']).time()  # saves record's time into a parameter.
# stadium = master_try.at[0,'stadium_temp']  # saves record's stadium into a parameter.
#
# kk = ligat_haal[(ligat_haal['match_date'] == d) & (ligat_haal['kot'] == t) &
#                     (ligat_haal['stadium'] == stadium)]['attendance'].values[0]
# print(0)





a = attendance(master_try.iloc[246])

master_try = master_try.apply(attendance, axis=1)
print(0)


def get_row(df_target, col, value):
    """
    to find the index number of the stadiums in the 'new' df.
    :param df_target:
    :param col:
    :param value:
    :return:
    """
    row_number = df_target[df_target[col] == value]
    return row_number.index[0]


def similar_option_2(master_name, i):
    """
    :param master_name: string. The name of stadiums from the master df.
    :return:
    """
    if is_in(master_name, new):  # for all the stadiums thar have the same name in 'master' as in 'new'
        row = get_row(new, 'Stadium', master_name)
        master.at[i, 'stadium_capacity'] = new.at[row, 'Capacity']
        master.at[i, 'stadium_cooradinates'] = new.at[row, 'Coordinates']
        master.at[i, 'stadium_age'] = 2023 - int(new.at[row, 'Built in'])
    else:
        if master_name == 'Teddi Malcha':
            row = get_row(new, 'Stadium', 'Teddy')
        elif master_name == 'Yaakov Turner Toto' or master_name == 'Toto Turner':
            row = get_row(new, 'Stadium', 'Turner')
        elif master_name == 'Winner':
            row = get_row(new, 'Stadium', 'Netanya')
        elif master_name == 'National Stadium Ramat':
            row = get_row(new, 'Stadium', 'Ramat Gan')
        else:
            row = get_row(new, 'Stadium', 'Kiryat Shmona Municipal')
        master.at[i, 'stadium_capacity'] = new.at[row, 'Capacity']
        master.at[i, 'stadium_cooradinates'] = new.at[row, 'Coordinates']
        master.at[i, 'stadium_age'] = 2023 - int(new.at[row, 'Built in'])


for i in range(len(master)):
    similar_option_2(master.at[i, 'stadium_temp'], i)


# try to fine similar score:

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


# ress = master['stadium_temp'].apply(similar, args=(stadiums['Stadium'],))
# print(master.at[3, 'stadiums'].split(' Stad')[0])
# print(stadiums.at[1, 'Stadium'].split(' Stad')[0])
# print(fuzz.partial_ratio(master.at[3, 'stadiums'].split(' Stadium')[0], stadiums.at[1, 'Stadium'].split(' Stadium')[0]))

# def matchin_score(a, b):
#     return fuzz.partial_ratio(a.split(' Stadium')[0], b.split(' Stadium')[0])
#
#
# matching_score = lambda x, y: fuzz.ratio(x, y)
# a = list(map(matching_score, master.at[0, 'stadiums'], stadiums['Stadium']))

print(0)
