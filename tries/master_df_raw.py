import pandas as pd

# List of all relavent only csv files to append together into one dataframe.
csv_files = ["season_2016.csv", "season_2017.csv", "season_2018.csv", "season_2019.csv", "season_2020.csv",
             "season_2021.csv"]

master_df = pd.DataFrame()
# Appending the data of the cvs files to a data frame

for file in csv_files:
    df_temp = pd.read_csv('data_tables/' + file)
    master_df = master_df.append(df_temp, ignore_index=True)

master_df = pd.read_csv(r'data_tables/master_data.csv')

# Renaming some of the columns for more meaningfull names
master_df.rename(
    columns={'id': 'gameID', 'id.1': 'venueID', 'name': 'stadiums', 'id.2': 'leagueID', 'name.1': 'leagueName',
             'id.3': 'homeTeamID',
             'name.2': 'homeTeamName', 'logo.1': 'homeTeamLogo', 'winner': 'homeWinner',
             'winner.1': 'awayWinner', 'id.4': 'awayTeamID', 'name.3': 'awayTeamName',
             'logo.2': 'awayTeamLogo', 'home.1': 'homeGoals', 'away.1': 'awayGoals',
             'home.2': 'homeHalfTimeScore', 'away.2': 'awayhomeHalfTimeScore',
             'home.3': 'homeFullTimeScore', 'away.3': 'awayFullTimeScore', 'home.4': 'homeExtraTimeScore',
             'away.4': 'awayExtraTimeScore', 'home.5': 'homePenaltyTimeScore',
             'away.5': 'awayPenaltyTimeScore', 'first': 'KOT'}, inplace=True)

headers_name = []  # A list with all of the headres names.
for col in master_df.columns:
    headers_name.append(col)


def delete_null_col(df, headers_names, x):
    """
    Delets null colmns from a data frame.
    :param df: DataFrame. That needed to be deleted from.
    :param headers_names: List. of the columns names.
    :param x: int. The number of the rows of the dataframe.
    """
    for name in headers_names:
        flag = True  # Checking if the coulms are empty ; True => empty(null) ; False => not empty(not null)
        for i in range(x):
            if pd.notnull(df.at[i, name]):  # not null => there are values
                flag = False
                break
        if flag:  # If the flag is True => the entire coulmn is Null so there is no information so we delete it.
            df.drop(columns=name, inplace=True)
            headers_names.remove(name)


def delete_None_col(df, headers_names, x):
    """
    Delets "None" colmns from a data frame.
    :param df: DataFrame. That needed to be deleted from.
    :param headers_names: List. of the columns names.
    :param x: int. The number of the rows of the dataframe.
    """
    for name in headers_names:
        flag = True  # Checking if the coulms are only with "None" ; True => all the rows ar "None" (justt None) ;
        # False => not all the rows ar "None" (other values).
        for i in range(x):  # Loop over all the rows of the data frame.
            if (df.at[i, name]) != "None":  # not None => there are other values
                flag = False
                break
        if flag:  # If the flag is True => the entire coulmn is "None" so there is no information so we delete it.
            df.drop(columns=name, inplace=True)
            headers_names.remove(name)


delete_null_col(master_df, headers_name, len(master_df))
delete_None_col(master_df, headers_name, len(master_df))

# Removing unnecessary columns.
master_df.drop(['Unnamed: 0', 'timezone', 'long', 'short', 'elapsed', 'home', 'halftime'], axis='columns', inplace=True)

# Changing the name "Hapoel Katamon" to "Hapoel Katamon Jerusalem"
master_df['homeTeamName'] = master_df['homeTeamName'].str.replace("Hapoel Katamon", "Hapoel Katamon Jerusalem")
master_df['awayTeamName'] = master_df['awayTeamName'].str.replace("Hapoel Katamon", "Hapoel Katamon Jerusalem")

# Saving the rew df into csv file.
# master_df.to_csv('master_df_raw.csv')
