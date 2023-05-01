# List of the beginning of each kot category (i.e. 15-16)
start = list(map(lambda x: datetime.strptime(x, '%H:%M:%S').time(),
                 ['15:00:00', '16:00:00', '17:00:00', '18:00:00', '19:00:00', '20:00:00', '21:00:00', '22:00:00']))
# List of the end of each kot category (i.e. 15-16)
end = list(map(lambda x: datetime.strptime(x, '%H:%M:%S').time(),
               ['16:00:00', '17:00:00', '18:00:00', '19:00:00', '20:00:00', '21:00:00', '22:00:00', '23:00:00']))
def kot_determining(t):
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


# Creating a temp column to categorize the kot
master_df['kot_temp'] = master_df['kot'].apply(kot_determining)
# Converting the values in the temp column into one-hot encoded vectors in a temp dataframe
temp_df = pd.get_dummies(master_df.kot_temp, prefix='kot')
master_df = pd.concat([master_df, temp_df], axis=1, join='inner')  # Concating the temp df into the master df
master_df.drop(['kot_temp'], axis='columns', inplace=True)  # Dropping the temp column