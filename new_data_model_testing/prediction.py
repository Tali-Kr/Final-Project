import math
from math import log, log10, exp
import pandas as pd
from xgboost_model.preprocessDf import preprocess
from xgboost_model.train_cols import train_cols
import joblib
from new_data_model_testing.duplicated_new_data import duplicate
from max_predictions import get_best_day

# Methods
def att(ar, cap, pred):
    cap = math.exp(cap)
    if pred == 0:
        ar = math.exp(ar)
        if ar > cap:
            ar = cap
        return ar/cap
    if pred == 1:
        if ar > 1:
            ar = 1
        if ar < 0:
            ar = ar*(-1)
        return math.log(cap*ar, math.e)

# region attendance

# Import Files to DF
df = pd.read_csv('../dataset_preparation/dt_prep_tables/pred_master_dataset.csv')
team_value_df = pd.read_csv('../web_scraping/teams_values.csv')
ln_attendance_orig = df['ln(attendance)']

# duplicate the df
df = duplicate(df)

# Preprocess the raw data
new_data = preprocess(df, team_value_df)
att_new_data = train_cols(new_data, 3)
att_new_data.drop(['ln(attendance)'], axis=1, inplace=True)

# load the model from file
att_model = joblib.load('../xgboost_model/ln_attendance.pkl')
df['ln(attendance)_pred'] = att_model.predict(att_new_data)

# adjusting the values
df[df['ln(attendance)_pred'] < 0] = 0
df['attendance_pred'] = df['ln(attendance)_pred'].apply(lambda x: math.exp(x))
df[df['attendance_pred'] == 1] = 0
df['ln(attendance)_orig'] = ln_attendance_orig

# getting the bast day for each game
att_pred = get_best_day(df, 3)

days_dic = {'0': 'Monday', '1': 'Tuesday', '2': 'Wednesday', '3': 'Thursday', '4': 'Friday', '5': 'Saturday',
            '6': 'Sunday'}  # Creating a dictionery to change the number of the day of the week to the name of the day.
att_pred['day_of_week'] = att_pred['day_of_week_num'].apply(lambda x: days_dic.get(x))

# Saving the results
# att_pred.to_csv('attendance_predic.csv')
#endregion
print()


# region home/away

# df = pd.read_csv('../dataset_preparation/dt_prep_tables/pred_master_dataset.csv')
# team_value_df = pd.read_csv('../xgboost_model/teams_values.csv')
# # ln_attendance_orig = df['ln(attendance)']
#
# # Preprocess the raw data
new_data = preprocess(df, team_value_df)
att_new_data = train_cols(new_data, 3)
att_new_data.drop(['ln(attendance)'], axis=1, inplace=True)


# load the model from file
att_model = joblib.load('../xgboost_model/ln_attendance.pkl')
new_data['ln(attendance)_pred'] = att_model.predict(att_new_data)

att_pred = get_best_day(new_data, 3)
att_pred['day_of_week'] = att_pred['day_of_week_num'].apply(lambda x: days_dic.get(x))
att_pred.to_csv("ln_att_predic.csv")

# duplicate the df
# df_dup = duplicate(new_data)
print()

# Choose the training columns 1 - Home win; 2 - Away win;
home_new_data = train_cols(new_data, 1)
home_new_data.drop(['home_won'], axis=1, inplace=True)
away_new_data = train_cols(new_data, 2)
away_new_data.drop(['away_won'], axis=1, inplace=True)

# load the model from file
who_win = [['home', home_new_data], ['away', away_new_data]]
for who in who_win:
    model = joblib.load('../xgboost_model/'+who[0]+'_won.pkl')
    new_data[who[0]+'_won_pred'] = model.predict(who[1])
    new_data[who[0]+'_won_pred_prob'] = model.predict_proba(who[1])[:, 1]


# getting the bast day for each game
home_pred = get_best_day(new_data, 1)  # home won
away_pred = get_best_day(new_data, 2)  # away won

# Saving the results
home_pred.drop(['away_won_pred', 'away_won_pred_prob'], axis=1, inplace=True)
home_pred['day_of_week'] = home_pred['day_of_week_num'].apply(lambda x: days_dic.get(x))
home_pred.to_csv("home_won_predic.csv")
away_pred.drop(['home_won_pred', 'home_won_pred_prob'], axis=1, inplace=True)
away_pred['day_of_week'] = away_pred['day_of_week_num'].apply(lambda x: days_dic.get(x))
away_pred.to_csv("away_won_predic.csv")

# endregion
print()