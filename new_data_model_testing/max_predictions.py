import pandas as pd


def get_max_prediction(df, code):
    """
    Gets the max prediction, according to the given code that is set to be the key in a dictionary.
    :param df: dataframe. Filterd df with all the option for the different day values.
    :param code: int. Which column (parameter) need to get the max value of.
    :return: series. Record of the max prediction.
    """
    best_day_options = {1: 'home_won_pred_prob', 2: 'away_won_pred_prob', 3: 'ln(attendance)_pred'}
    # Sets the option that was given, and will be the name of the column to extract the max value of.
    opt = best_day_options.get(code)
    res = pd.DataFrame(df[df[opt] == max(df[opt])])  # Filter the given df to the max value.
    return res  # Return a record (series) of the max value.


def get_best_day(df, code):
    """
    Creates new data frame of the best optional day for the give paramer (code).
    :param df: dataframe. The data that need to extract the best option for each game.
    :param code: int. Which column (parameter) need to get the max value of.
    :return: dataframe. For each game what is the best day to play based on the paramer that was given.
    """
    unique_key = list((df['home_key'].unique()))  # Df that saves the unique keys for each game.

    res = []  # List to store the resulting series objects

    for key in unique_key:
        game_df = df[df['home_key'] == key]  # Filter the dataframe for each game key.
        max_prediction = get_max_prediction(game_df, code)  # Get the max prediction for the game.
        if len(max_prediction) > 1:
            max_prediction = max_prediction.drop_duplicates(subset='Unnamed: 0')
        res.append(max_prediction)  # Append the resulting series to the list.

    res_df = pd.concat(res).reset_index(drop=True)  # Concatenate the list of series into a dataframe.
    res_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    return res_df
