import math
import random
import statsmodels.api as sm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import category_encoders as ce
import xgboost
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import normalize
from sklearn.impute import KNNImputer
from sklearn import tree
from tqdm import tqdm
from xgboost import XGBClassifier, plot_importance, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from preprocessDf import preprocess

# region Import Tables
df = pd.read_csv('../dataset_preparation/dt_prep_tables/master_dataset.csv')
team_value_df = pd.read_csv('teams_values.csv')
#endregion

# Preprocess df
df = preprocess(df, team_value_df)
print()
# train_cols = ['round', 'kot',  'away_won',
#                   'home_pos_b4_game', 'home_lg_b4_game', 'away_pos_b4_game', 'away_lg_b4_game',
#                   'kot_15_16', 'kot_16_17', 'kot_17_18', 'kot_18_19', 'kot_19_20','kot_20_21', 'kot_21_22',
#                   'kot_22_23', 'derby', 'while_champion_league', 'while_european_games', 'home_promoted','home_is_relegated',
#                   'home_is_champion', 'away_promoted', 'away_is_relegated', 'away_is_champion',
#                   'att_ratio', 'ln(attendance)', 'stadium_age', 'distance_in_km',  'day_of_week_num',  'month',
#                   'home_value', 'away_value', 'home_underdog', 'away_underdog']
# print(len(train_cols))
# cols = df.columns.values.tolist()
# print(cols)
# choose the columns that needed to prediction
prediction = int(input('Enter: \nHome win = 0\nAway win = 1\n'))

# region training columns define and Train split
#region underdog prediction
# if prediction == 0:
#     # Choose training columns underdog
#     train_cols = ['round_type', 'round',   'kot_15_16', 'kot_16_17', 'kot_17_18', 'kot_18_19',
#                   'kot_19_20', 'kot_20_21', 'kot_21_22', 'kot_22_23', 'derby',
#                   'while_world_cup', 'while_champion_league', 'while_european_games',
#                   'home_promoted',  'home_is_champion', 'away_promoted',
#                   'away_is_champion', 'att_ratio', 'ln(attendance)',
#                   'ln(capacity)', 'stadium_age', 'distance_in_km', 'day_of_week_num',
#                   'home_value', 'away_value', 'udw'] #,'home_underdog', 'away_underdog']
#     # deleted = ['home_team', 'away_team','home_position','home_league_pts','away_league_pts',
#     #               'away_position',]
#     # all_columns =  ['date', 'kot', 'season', 'round_type', 'round', 'home_team', 'home_team_city', 'away_team', 'away_team_city',
#     #  'referee', 'attendance', 'home_score', 'away_score', 'home_won', 'away_won', 'draw', 'home_pos_b4_game',
#     #  'home_lg_b4_game', 'away_pos_b4_game', 'away_lg_b4_game', 'underdog', 'home_position', 'away_position',
#     #  'kot_15_16', 'kot_16_17', 'kot_17_18', 'kot_18_19', 'kot_19_20', 'kot_20_21', 'kot_21_22', 'kot_22_23', 'derby',
#     #  'while_world_cup', 'while_champion_league', 'while_european_games', 'home_promoted', 'home_league_pts',
#     #  'home_is_relegated', 'home_is_champion', 'away_promoted', 'away_league_pts', 'away_is_relegated',
#     #  'away_is_champion', 'game_stadium', 'capacity', 'att_ratio', 'ln(attendance)', 'ln(capacity)', 'built_in',
#     #  'stadium_age', 'stadium_age_squared', 'home_team_stadium', 'away_team_stadium', 'distance_in_km', 'home_key',
#     #  'away_key', 'day_of_week_num', 'day_of_week', 'month', 'udw', 'home_underdog', 'away_underdog', 'home_value',
#     #  'home_squad_s', 'away_value', 'away_squad_s']
#
#     #region talis 1st try:
#     train_cols = ['round', 'kot_15_16', 'kot_16_17', 'kot_17_18', 'kot_18_19',
#                   'kot_19_20', 'kot_20_21', 'kot_21_22', 'kot_22_23', 'derby',
#                   'while_champion_league', 'while_european_games',
#                   'home_promoted', 'home_is_champion', 'away_promoted',
#                   'away_is_champion', 'att_ratio', 'ln(attendance)',
#                   'ln(capacity)',  'distance_in_km', 'day_of_week_num',
#                   'home_value', 'away_value', 'udw', 'home_underdog', 'away_underdog']
#     #endregion
#     # deleted = ['home_team', 'away_team','home_position','home_league_pts','away_league_pts',
#     #               'away_position',]
#     # 1st try: deleted = ['round_type','while_world_cup','stadium_age',
#     # region PLEASE HELP ME ROMAN !!!!!!!!!!!!!!!!
#     # def unique(list1):
#     #
#     # # initialize a null list
#     #     unique_list = []
#     #
#     # # traverse for all elements
#     #     for x in list1:
#     #     # check if exists in unique_list or not
#     #         if x not in unique_list:
#     #             unique_list.append(x)
#     #     # print list
#     #     return unique_list
#     # def generate_unique_numbers(k, lst):
#     #     random.seed(53)
#     #     res_lst = []
#     #     for i in range(0,k):
#     #         unq_rndm = random.choice(lst)
#     #         lst.remove(unq_rndm)
#     #         res_lst.append(unq_rndm)
#     #     return res_lst
#     #
#     #
#     train_data = df[train_cols].copy(deep=True)
#     #
#     # won = train_data[train_data['udw'] == 1]
#     # not_won = train_data[train_data['udw'] == 0]
#     #
#     # won_index = list(won.index.values)
#     # ran_numbers = generate_unique_numbers(317, won_index)
#     # won_df = pd.DataFrame()
#     #
#     #
#     # for num in ran_numbers:
#     #     won_row = train_data.iloc[num].to_frame().T
#     #     frames = [won_df, won_row]
#     #     won_df = pd.concat(frames)
#     #
#     # frames = [won_df,train_data]
#     # train_data = pd.concat(frames)
#     #
#     # # not_won = train_data[train_data['udw'] == 0]
#     # not_won_index = list(not_won.index.values)
#     # ran_numbers = generate_unique_numbers(317, not_won_index)
#     # no_won_df = pd.DataFrame()
#     #
#     # # a = train_data.copy(deep=True)
#     #
#     # for num in ran_numbers:
#     #     a = num
#     #     b = len(train_data.index)
#     #     if len(train_data.index) > 2400:
#     #         train_data = train_data.drop(labels=train_data.index[num])
#     #     else:
#     #         break
#     # # train_data = train_data.reset_index()
#     # # train_data = train_data.drop('index', axis=1)
#     # # frames = [won_df_res, a]
#     # # train_data = pd.concat(frames)
#     #
#     # won = train_data[train_data['udw'] == 1]
#     # not_won = train_data[train_data['udw'] == 0]
#     #
#     # print()
#     # endregion
#     # Train Split
#     X = train_data.drop('udw', axis=1)
#     y = train_data['udw']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#endregion

#region home prediction
if prediction == 0:
    # Choose training columns home won
    # train_cols = ['month', 'round', 'home_pos_b4_game', 'away_pos_b4_game', 'kot', 'derby',
    #               'while_champion_league', 'while_european_games', 'home_lg_b4_game',
    #               'home_is_champion', 'away_lg_b4_game', 'away_is_champion', 'att_ratio',
    #               'ln(attendance)', 'ln(capacity)', 'stadium_age', 'distance_in_km', 'day_of_week_num', 'home_value',
    #               'away_value', 'home_won']
    #region talis 1st try: 69% : confussin matrix [[260 29] [119 72]] XXXX
    # train_cols = ['round', 'kot',  'home_won',
    #               'home_pos_b4_game', 'home_lg_b4_game', 'away_pos_b4_game', 'away_lg_b4_game',
    #               'kot_15_16', 'kot_16_17', 'kot_17_18', 'kot_18_19', 'kot_19_20','kot_20_21', 'kot_21_22',
    #               'kot_22_23', 'derby', 'while_champion_league', 'while_european_games', 'home_is_relegated',
    #               'home_is_champion', 'away_promoted', 'away_league_pts', 'away_is_relegated', 'away_is_champion',
    #               'att_ratio', 'ln(attendance)', 'stadium_age', 'distance_in_km',  'day_of_week_num',  'month',
    #               'home_value', 'away_value']
    #endregion
    #region talis 2nd try: 69% confussin matrix [[261 28] [120 71]] ; XXX wo 69% 'away_promoted' confussin matrix [[261 28] [119 72]] XXX
    # train_cols = ['round', 'kot', 'home_won', 'home_pos_b4_game', 'home_lg_b4_game', 'away_pos_b4_game',
    #               'away_lg_b4_game', 'derby', 'home_is_relegated', 'home_is_champion',
    #               'away_league_pts', 'away_is_relegated', 'away_is_champion', 'att_ratio', 'ln(attendance)',
    #               'stadium_age', 'distance_in_km', 'day_of_week_num', 'month', 'home_value', 'away_value']
    # endregion
    # region talis 3rd try: 70% confussin matrix [[261 28] [120 71]] ; wo  68%  'away_league_pts' confussin matrix [[260 29] [123 68]] XXXX
    # train_cols = ['round', 'kot', 'home_won', 'home_pos_b4_game', 'home_lg_b4_game', 'away_pos_b4_game',
    #               'away_lg_b4_game', 'derby', 'home_is_relegated', 'home_is_champion',
    #               'away_is_relegated', 'away_is_champion', 'att_ratio', 'ln(attendance)',
    #               'day_of_week_num', 'month', 'home_value', 'away_value']
    # endregion
    # region talis 4th try: 68% confussin matrix [[258 31] [123 68]]
    # train_cols = ['round', 'kot', 'home_won', 'home_pos_b4_game', 'home_lg_b4_game', 'away_pos_b4_game',
    #               'away_lg_b4_game', 'att_ratio', 'ln(attendance)',
    #               'day_of_week_num', 'month', 'home_value', 'away_value']
    # endregion
    # region talis 5th try: 69%  confussin matrix [[261 28] [123 68]]
    # train_cols = ['round', 'kot',  'home_won',
    #               'home_pos_b4_game', 'home_lg_b4_game', 'away_pos_b4_game', 'away_lg_b4_game',
    #               'kot_15_16', 'kot_16_17', 'kot_17_18', 'kot_18_19', 'kot_19_20','kot_20_21', 'kot_21_22',
    #               'kot_22_23', 'derby', 'while_champion_league', 'while_european_games', 'home_is_relegated',
    #               'home_is_champion', 'away_promoted', 'away_is_relegated', 'away_is_champion',
    #               'att_ratio', 'ln(attendance)', 'stadium_age', 'distance_in_km',  'day_of_week_num',  'month',
    #               'home_value', 'away_value', 'home_underdog', 'away_underdog']
    # endregion
    # region talis 6th try: 69%  confussin matrix [[260 29] [122 69]] *****************
    train_cols = ['round', 'kot', 'home_won',
                  'home_pos_b4_game', 'home_lg_b4_game', 'away_pos_b4_game', 'away_lg_b4_game',
                  'kot_18_19', 'kot_19_20', 'kot_20_21',   'while_european_games', 'home_is_relegated',
                  'home_is_champion', 'away_promoted', 'away_is_relegated', 'away_is_champion',
                  'att_ratio', 'ln(attendance)', 'stadium_age', 'distance_in_km', 'day_of_week_num', 'month',
                  'home_value', 'away_value', 'home_underdog', 'away_underdog']
    # endregion
    # region talis 7th try: 68%  confussin matrix [[258 31] [122 69]]
    # train_cols = ['round', 'kot', 'home_won',
    #               'home_pos_b4_game', 'home_lg_b4_game', 'away_pos_b4_game', 'away_lg_b4_game',
    #               'kot_18_19', 'kot_19_20', 'kot_20_21', 'while_european_games',  'away_promoted',
    #               'att_ratio', 'ln(attendance)', 'stadium_age', 'distance_in_km', 'day_of_week_num', 'month',
    #               'home_value', 'away_value', 'home_underdog', 'away_underdog']
    # endregion
    # region talis 8th try: 69% confussin matrix [[259 30] [120 71]]
    # train_cols = ['round', 'kot', 'home_won', 'home_pos_b4_game', 'home_lg_b4_game', 'away_pos_b4_game',
    #               'away_lg_b4_game', 'derby', 'home_is_relegated', 'home_is_champion', 'home_underdog', 'away_underdog',
    #               'away_league_pts', 'away_is_relegated', 'away_is_champion', 'att_ratio', 'ln(attendance)',
    #               'stadium_age', 'distance_in_km', 'day_of_week_num', 'month', 'home_value', 'away_value']
    # endregion
    # region talis 9th try: 69% confussin matrix [[259 30] [119 72]]
    # train_cols = ['round', 'kot', 'home_won', 'home_pos_b4_game', 'home_lg_b4_game', 'away_pos_b4_game',
    #               'away_lg_b4_game', 'home_underdog', 'away_underdog',
    #               'away_league_pts', 'att_ratio', 'ln(attendance)',
    #               'stadium_age', 'distance_in_km', 'day_of_week_num', 'month', 'home_value', 'away_value']
    # endregion


    #region deleted columns
    # deleted = ['date',  'season', 'round_type','home_team', 'home_team_city', 'away_team', 'underdog',
    #             'away_team_city', 'referee', 'home_score', 'away_score','away_won', 'draw','home_position',
    #            'away_position', 'while_world_cup','home_promoted', 'home_league_pts','game_stadium','capacity',
    #            'attendance', 'ln(capacity)', 'built_in','stadium_age_squared', 'home_team_stadium', 'away_team_stadium',
    #            'home_key', 'away_key','day_of_week', 'home_squad_s','udw', 'away_squad_s']
    # 2nd try: deleted [ 'kot_15_16', 'kot_16_17', 'kot_17_18', 'kot_18_19', 'kot_19_20', 'kot_20_21', 'kot_21_22',
    #                   'kot_22_23','while_champion_league', 'while_european_games','away_promoted']
    # 3rd try: deleted ['away_promoted','stadium_age', 'distance_in_km', 'away_league_pts' ]
    # 4th try: deleted ['derby', 'away_league_pts','home_is_relegated', 'home_is_champion',
    #     #                'away_is_relegated', 'away_is_champion', ]
    # 5th try: deleted = ['date',  'season', 'round_type','home_team', 'home_team_city', 'away_team', 'underdog',
    #             'away_team_city', 'referee', 'home_score', 'away_score','away_won', 'draw','home_position',
    #            'away_position', 'while_world_cup','home_promoted', 'home_league_pts','away_league_pts','game_stadium','capacity',
    #            'attendance', 'ln(capacity)', 'built_in','stadium_age_squared', 'home_team_stadium', 'away_team_stadium',
    # 6th try: deleted = [ 'kot_15_16', 'kot_16_17', 'kot_17_18','kot_21_22',
    #                   'kot_22_23','derby', 'while_champion_league',
    # 7th try: deleted = ['home_is_relegated','home_is_champion','away_is_relegated', 'away_is_champion']
    # 9h try: deleted = [ 'derby', 'home_is_relegated', 'home_is_champion','away_is_relegated', 'away_is_champion', ]
    #endregion

    train_data = df[train_cols].copy(deep=True)
    # Train Split
    X = train_data.drop('home_won', axis=1)
    y = train_data['home_won']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#endregion

#region away prediction
if prediction == 1:
    # Choose training columns away won
    # train_cols = ['round', 'home_pos_b4_game', 'month', 'derby',
    #               'away_pos_b4_game', 'kot', 'while_european_games', 'home_promoted', 'away_promoted', 'home_lg_b4_game',
    #               'home_is_champion', 'away_lg_b4_game', 'att_ratio', 'ln(attendance)', 'ln(capacity)', 'stadium_age',
    #               'distance_in_km', 'day_of_week_num', 'home_value', 'away_value', 'away_won']

    #region tali 1st try: 74%  confussin matrix [[316 16] [109 39]] ******************
    train_cols = ['round', 'kot',  'away_won',
                  'home_pos_b4_game', 'home_lg_b4_game', 'away_pos_b4_game', 'away_lg_b4_game',
                  'kot_15_16', 'kot_16_17', 'kot_17_18', 'kot_18_19', 'kot_19_20','kot_20_21', 'kot_21_22',
                  'kot_22_23', 'derby', 'while_champion_league', 'while_european_games', 'home_promoted','home_is_relegated',
                  'home_is_champion', 'away_promoted', 'away_is_relegated', 'away_is_champion',
                  'att_ratio', 'ln(attendance)', 'stadium_age', 'distance_in_km',  'day_of_week_num',  'month',
                  'home_value', 'away_value', 'home_underdog', 'away_underdog']
    #endregion
    # region tali 2nd try: 74%  confussin matrix [[316 16] [110 38]]
    # train_cols = ['round', 'kot', 'away_won', 'home_pos_b4_game', 'home_lg_b4_game', 'away_pos_b4_game',
    #               'away_lg_b4_game', 'kot_17_18', 'kot_18_19', 'kot_19_20', 'kot_20_21', 'kot_21_22',
    #               'home_promoted', 'att_ratio', 'ln(attendance)', 'stadium_age', 'distance_in_km',
    #               'day_of_week_num', 'month', 'home_value', 'away_value', 'home_underdog', 'away_underdog']
    #endregion
    # region tali 3rd try: 73%  confussin matrix [[315 17] [112 36]]
    # train_cols = ['round', 'kot', 'away_won', 'home_pos_b4_game', 'home_lg_b4_game', 'away_pos_b4_game',
    #               'away_lg_b4_game',
    #               'home_promoted', 'att_ratio', 'ln(attendance)', 'stadium_age', 'distance_in_km',
    #               'day_of_week_num', 'month', 'home_value', 'away_value', 'home_underdog', 'away_underdog']
    # endregion
    # region deleted columns
    #talis 1st try: deleted = ['date',  'season', 'round_type','home_team', 'home_team_city', 'away_team',
    # 'away_team_city', 'referee', 'home_score', 'away_score', 'draw','home_won','home_position',
    #            'away_position', 'while_world_cup', 'home_league_pts','game_stadium','capacity',
    #            'attendance', 'ln(capacity)', 'built_in','stadium_age_squared', 'home_team_stadium', 'away_team_stadium',
    #            'home_key', 'away_key','day_of_week', 'home_squad_s','udw', 'away_squad_s', 'underdog', 'away_league_pts']
    # talis 2nd try: deleted = ['while_champion_league', 'while_european_games','home_is_relegated',
    #                   'home_is_champion', 'away_promoted', 'away_is_relegated', 'away_is_champion','kot_15_16', 'kot_16_17','derby','kot_22_23',]
    #talis 3rd try: deleted = [ 'kot_17_18', 'kot_18_19', 'kot_19_20', 'kot_20_21', 'kot_21_22']
    #endregion

    train_data = df[train_cols].copy(deep=True)
    # Train Split
    X = train_data.drop('away_won', axis=1)
    y = train_data['away_won']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# endregion
# endregion

# region XGBoost test and tune hyperparameters

# # Define classifier
# xgb_model = XGBClassifier()

# # Define the hyperparameters to search over
# params = {
#     'learning_rate': [x for x in np.linspace(start=0.01, stop=0.2, num=4)],
#     'max_depth': range(1, 10, 1),
#     'min_child_weight': range(1, 13, 2),
#     'gamma': [i/10.0 for i in range(0, 5)],
#     'subsample': [i/100.0 for i in range(60, 100, 5)],
#     'colsample_bytree': [i/100.0 for i in range(60, 100, 5)],
#     'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05],
#     'eta' : 0.3
# }

# # Define the grid search object
# grid_search = GridSearchCV(
#     estimator=xgb_model,
#     param_grid=params,
#     cv=5,   # 5-fold cross validation
#     scoring='accuracy',
#     n_jobs=-1
# )

# # Fit the grid search object to the training data
# grid_search.fit(X_train, y_train)

# # Print the best score and the best hyperparameters for the model
# print(grid_search.best_score_)
# print(grid_search.best_params_)

# # Get the best estimator from the grid search
# best_xgb_model = grid_search.best_estimator_

# # Accuracy on Train
# train_accuracy = best_xgb_model.score(X_train, y_train) * 100
# print(f'Training Accuracy is: {train_accuracy:.2f}%')

# # Accuracy on Test
# test_accuracy = best_xgb_model.score(X_test, y_test) * 100
# print(f'Testing Accuracy is: {test_accuracy:.2f}%')

# # Test predictions
# xgb_model_pred = best_xgb_model.predict(X_test)
# print('Win', sum(xgb_model_pred != 0))
# print('Lose', sum(xgb_model_pred == 0))

# # Classification report and confusion matrix
# print(classification_report(y_test, xgb_model_pred))
# print(confusion_matrix(y_test, xgb_model_pred))

# # Feature importance plot
# plot_importance(best_xgb_model)
# plt.show()

# # plot learning curve to see if the model is overfitted
# from sklearn.model_selection import learning_curve
# import matplotlib.pyplot as plt

# train_sizes, train_scores, test_scores = learning_curve(best_xgb_model, X_train, y_train, cv=5,
#                                                         train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')
#
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)

# plt.figure()
# plt.title("Learning Curve")
# plt.xlabel("Training Examples")
# plt.ylabel("Accuracy Score")
# plt.ylim((0.5, 1.01))
# plt.grid()

# plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
# plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
# plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
# plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
# plt.legend(loc="best")
# plt.show()

# endregion

# region Define the XGBoost model
if prediction == 0:  # Home won
    best_xgb_model = XGBClassifier(learning_rate=0.01, max_depth=5, min_child_weight=9, gamma=0.4,
                                   subsample=0.6, reg_alpha=0, colsample_bytree=0.75)
#     {'colsample_bytree': 0.75, 'gamma': 0.4, 'learning_rate': 0.01, 'max_depth': 5, 'min_child_weight': 9, 'reg_alpha': 0, 'subsample': 0.6}
if prediction == 1:  # Away Won
    best_xgb_model = XGBClassifier(learning_rate=0.01, max_depth=5, min_child_weight=9, gamma=0.4,
                                   subsample=0.6, reg_alpha=0, colsample_bytree=0.75)
#     learning_rate=0.046, max_depth=3, min_child_weight=0.3, gamma=0.001,subsample=0.707, reg_alpha=0.001
# endregion

# region Fit the classifier object to the training data
best_xgb_model.fit(X_train, y_train)
# endregion

# Save model
pred_type = ['home_won', 'away_won']
joblib.dump(best_xgb_model, pred_type[prediction]+'.pkl')

if prediction == 0:
    dropping_col = 'home_won'
if prediction == 1:
    dropping_col = 'away_won'

for i, column in enumerate(train_data.drop(dropping_col, axis=1)):
    # print('Importance of feature {}:, {:.3f}'.format(column, best_xgb_model.feature_importances_[i]))

    fi = pd.DataFrame({'Variable': [column], 'Feature Importance Score': [best_xgb_model.feature_importances_[i]]})

    try:
        final_fi = pd.concat([final_fi, fi], ignore_index=True)
    except:
        final_fi = fi

# Ordering the data
final_fi = final_fi.sort_values('Feature Importance Score', ascending=False).reset_index()
print(final_fi)

# region Save the model
# if prediction == 0:
#     joblib.dump(best_xgb_model, 'home_win.pkl')
# if prediction == 1:
#     joblib.dump(best_xgb_model, 'away_win.pkl')
# endregion

# region Accuracy on Train
train_accuracy = best_xgb_model.score(X_train, y_train) * 100
print(f'Training Accuracy is: {train_accuracy:.2f}%')
# endregion

# region Accuracy on Test
test_accuracy = best_xgb_model.score(X_test, y_test) * 100
print(f'Testing Accuracy is: {test_accuracy:.2f}%')
# endregion

# region Test prediction
xgb_model_pred = best_xgb_model.predict(X_test)
print('Win', sum(xgb_model_pred != 0))
print('Lose', sum(xgb_model_pred == 0))
print(classification_report(y_test, xgb_model_pred))
print(confusion_matrix(y_test, xgb_model_pred))
# endregion

# pd.DataFrame(xgb_model_pred).to_csv("home_prediction.csv")

print()

# region Feature importance plot
if prediction == 0:
    path = '../../project_BOOK/models_plots/home/Feature importance_BOOK.png'
if prediction == 1:
    path = '../../project_BOOK/models_plots/away/Feature importance_BOOK.png'
plot_importance(best_xgb_model, max_num_features=20, color='#258B9E', grid=False, height=0.65)

plt.title("Feature importance", fontsize=18)
plt.xlabel("F Score", fontsize=14)
plt.ylabel("Features", fontsize=14)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

fig = plt.gcf()
fig.set_size_inches(1920/100, 1080/100)
plt.savefig(path)
plt.show()
# endregion
print()
# region Plot learning curve to see if the model is overfitted
# from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, test_scores = learning_curve(best_xgb_model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

if prediction == 0:
    path = '../../project_BOOK/models_plots/home/Learning Curve_BOOK.png'
if prediction == 1:
    path = '../../project_BOOK/models_plots/away/Learning Curve.png'

plt.figure()
plt.title("Learning Curve", fontsize=18)
plt.xlabel("Training Examples", fontsize=16)
plt.ylabel("Accuracy Score", fontsize=16)
plt.ylim((0, 1.01))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='#258B9E')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='#732C91')
plt.plot(train_sizes, train_scores_mean, 'o-', color='#258B9E', label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color='#732C91', label="Cross-validation score")

plt.legend(loc="best")
fig = plt.gcf()
fig.set_size_inches(1920/100, 1080/100)
plt.savefig(path)
plt.show()

print()
# https://stats.stackexchange.com/questions/438632/assessing-overfitting-via-learning-curves - answer to someone who asked it the model is good.
# https://scikit-learn.org/stable/modules/learning_curve.html - more about learning curves.
# endregion