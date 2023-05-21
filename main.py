import math
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
from tqdm import tqdm
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from preprocessDf import preprocess

# region Import Tables
df = pd.read_csv(r'table/master_data__new__08_05.csv')
team_value_df = pd.read_csv(r'table/teams_values.csv')
#endregion

# Preprocess df
df = preprocess(df, team_value_df)
print()

# choose the columns that needed to prediction
prediction = int(input('Enter: Underdog = 0\nHome win = 1\nAway win = 2\n'))

# region training columns define and Train split
if prediction == 0:
    # Choose training columns underdog
    train_cols = ['round_type', 'round', 'home_team', 'away_team', 'home_position',
                  'away_position', 'kot_15_16', 'kot_16_17', 'kot_17_18', 'kot_18_19',
                  'kot_19_20', 'kot_20_21', 'kot_21_22', 'kot_22_23', 'derby',
                  'while_world_cup', 'while_champion_league', 'while_european_games',
                  'home_promoted', 'home_league_pts', 'home_is_champion', 'away_promoted',
                  'away_league_pts', 'away_is_champion', 'att_ratio', 'ln(attendance)',
                  'ln(capacity)', 'stadium_age', 'distance_in_km', 'day_of_week_num',
                  'home_value', 'away_value', 'udw']
    train_data = df[train_cols].copy(deep=True)
    # Train Split
    X = train_data.drop('udw', axis=1)
    y = train_data['udw']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if prediction == 1:
    # Choose training columns home won
    train_cols = ['month', 'round_type', 'round', 'home_pos_b4_game', 'away_pos_b4_game', 'kot', 'derby',
                  'while_champion_league', 'while_european_games', 'home_promoted', 'home_lg_b4_game',
                  'home_is_champion', 'away_promoted', 'away_lg_b4_game', 'away_is_champion', 'att_ratio',
                  'ln(attendance)', 'ln(capacity)', 'stadium_age', 'distance_in_km', 'day_of_week_num', 'home_value',
                  'away_value', 'home_won']
    train_data = df[train_cols].copy(deep=True)
    # Train Split
    X = train_data.drop('home_won', axis=1)
    y = train_data['home_won']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if prediction == 2:
    # Choose training columns away won
    train_cols = ['round', 'home_pos_b4_game', 'month', 'derby',
                  'away_pos_b4_game', 'kot', 'while_european_games', 'home_promoted', 'away_promoted', 'home_lg_b4_game',
                  'home_is_champion', 'away_lg_b4_game', 'att_ratio', 'ln(attendance)', 'ln(capacity)', 'stadium_age',
                  'distance_in_km', 'day_of_week_num', 'home_value', 'away_value', 'away_won']
    train_data = df[train_cols].copy(deep=True)
    # Train Split
    X = train_data.drop('away_won', axis=1)
    y = train_data['away_won']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# endregion

# region XGBoost test and tune hyperparameters

# # Define classifier
# xgb_model = XGBClassifier()

# # Define the hyperparameters to search over
# params = {
#     # 'learning_rate': [x for x in np.linspace(start=0.01, stop=0.2, num=4)],
#     # 'max_depth': range(1, 10, 1),
#     # 'min_child_weight': range(1, 13, 2),
#     # 'gamma': [i/10.0 for i in range(0, 5)],
#     # 'subsample': [i/100.0 for i in range(60, 100, 5)],
#     # 'colsample_bytree': [i/100.0 for i in range(60, 100, 5)],
#     # 'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05]
# }

# # Define the grid search object
# grid_search = GridSearchCV(
#     estimator=xgb_model,
#     param_grid=params,
#     cv=5,   # 5-fold cross validation
#     scoring='accuracy',
#     verbose=2
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

# region XGBoost
# region Define the XGBoost model
if prediction == 0:  # Underdog
    best_xgb_model = XGBClassifier()

if prediction == 1:  # Home won
    best_xgb_model = XGBClassifier(learning_rate=0.01, max_depth=1, min_child_weight=0.1, gamma=0.01,
                                   subsample=0.18, reg_alpha=0.01)

if prediction == 2:  # Away Won
    best_xgb_model = XGBClassifier(learning_rate=0.046, max_depth=3, min_child_weight=0.3, gamma=0.001,
                                   subsample=0.707, reg_alpha=0.001)
# endregion

# region Fit the classifier object to the training data
best_xgb_model.fit(X_train, y_train)
# endregion

# region Save the model
# if prediction == 0:
#     joblib.dump(best_xgb_model, 'undedog_win.pkl')
# if prediction == 1:
#     joblib.dump(best_xgb_model, 'home_win.pkl')
# if prediction == 2:
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

# region Feature importance plot
plot_importance(best_xgb_model)
plt.show()
# endregion

# region Plot learning curve to see if the model is overfitted
# from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, test_scores = learning_curve(best_xgb_model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title("Learning Curve")
plt.xlabel("Training Examples")
plt.ylabel("Accuracy Score")
plt.ylim((0.5, 1.01))
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.legend(loc="best")
plt.show()
# https://stats.stackexchange.com/questions/438632/assessing-overfitting-via-learning-curves - answer to someone who asked it the model is good.
# https://scikit-learn.org/stable/modules/learning_curve.html - more about learning curves.
# endregion

# endregion