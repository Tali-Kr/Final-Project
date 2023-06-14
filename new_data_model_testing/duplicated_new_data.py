import pandas as pd

master_df = pd.read_csv("../dataset_preparation/dt_prep_tables/pred_master_dataset.csv")

# region most frequented kot
kot_freq = master_df.pivot_table(index='kot', aggfunc={'home_key': 'count'})
kot_freq = kot_freq.sort_values(by=['home_key'], ascending=False)
top_kot_freq = list(kot_freq.index.values)
kot = top_kot_freq[:6]
kot.insert(4, '19:30')  # To skip the 5th index.
# endregion


def duplicate(df):
    # Duplicates the records 6 times and changing the day of the week.
    res = pd.DataFrame()
    for i in range(0, 7):
        if i == 4:  # The model didnt have any train and test records with Friday.
            continue
        else:
            df_temp = df.assign(day_of_week_num=i)  # Sets the day of the week to be i in all of the records.
            frames = [df_temp, res]
            res = pd.concat(frames)  # Concate all the temp df into one df.
    return res


master_df = duplicate(master_df).reset_index()  # Reset the indeics of the df.
master_df.drop(['index', 'Unnamed: 0'], axis='columns', inplace=True)

# Creating a dictionery to change the number of the day of the week to the name of the day.
days_dic = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
master_df['day_of_week'] = master_df['day_of_week_num'].apply(lambda x: days_dic.get(x))

master_df.to_csv('dup_new_data.csv')
print()