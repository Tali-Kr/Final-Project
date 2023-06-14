import math
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessDf import preprocess
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, learning_curve
from xgboost import XGBRegressor, plot_importance

# region Functions ### do we need these functions ? there is no use to them
def expoRation(att, cap):
    return math.exp(att) / math.exp(cap)


def attBcap(att, cap):
    if att > cap:
        return cap
    else:
        return att


# endregion

# region Import table into DF
df = pd.read_csv('../dataset_preparation/dt_prep_tables/master_dataset.csv')  # Original final dataset
df_value = pd.read_csv('../web_scraping/teams_values.csv')
# endregion

# Preprocess data
df = preprocess(df, df_value)

#region What columns will be used for training

# region all numeric cols
cols = ['kot', 'season', 'round_type', 'round', 'attendance', 'home_score', 'away_score', 'home_won', 'away_won', 'draw',
        'home_pos_b4_game', 'home_lg_b4_game', 'away_pos_b4_game', 'away_lg_b4_game', 'home_position', 'away_position',
        'kot_15_16', 'kot_16_17', 'kot_17_18', 'kot_18_19', 'kot_19_20', 'kot_20_21', 'kot_21_22', 'kot_22_23', 'derby',
        'while_world_cup', 'while_champion_league', 'while_european_games', 'home_promoted', 'home_league_pts',
        'home_is_relegated', 'home_is_champion', 'away_promoted', 'away_league_pts', 'away_is_relegated',
        'away_is_champion', 'capacity', 'att_ratio', 'ln(attendance)', 'ln(capacity)', 'built_in',
        'stadium_age', 'stadium_age_squared', 'distance_in_km', 'day_of_week_num', 'month', 'udw', 'home_underdog',
        'away_underdog', 'home_value', 'home_squad_s', 'away_value', 'away_squad_s']
#endregion

train_cols = ['kot', 'season', 'round_type', 'round',
              'home_pos_b4_game', 'home_lg_b4_game', 'away_pos_b4_game', 'away_lg_b4_game',
              'kot_16_17', 'kot_17_18', 'kot_18_19', 'kot_19_20', 'kot_20_21', 'derby', 'while_champion_league',
              'while_european_games', 'home_promoted', 'home_is_relegated', 'home_is_champion', 'away_promoted',
              'away_is_relegated', 'away_is_champion', 'ln(attendance)', 'ln(capacity)',
              'stadium_age', 'distance_in_km', 'day_of_week_num', 'month',
              'home_underdog', 'away_underdog', 'home_value']
# endregion

# train data
train_data = df[train_cols].copy(deep=True)

# region X, y split
X = train_data.drop('ln(attendance)', axis=1)
y = train_data['ln(attendance)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# endregion

# region XGBoost test and tune hyperparameters
# xgb_model = XGBRegressor()

# Define the parameter grid to search
# param_grid = {
#     'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
#     'max_depth': [3, 4, 5, 6, 7],
#     'n_estimators': [50, 100, 150, 200, 250],
#     'reg_alpha': [0, 0.01, 0.1, 1, 10],
#     'reg_lambda': [0, 0.01, 0.1, 1, 10],
# }
# region Create an instance of GridSearchCV to perform the search
# grid_search = GridSearchCV(
#     estimator=xgb_model,
#     param_grid=param_grid,
#     cv=5,
#     scoring='r2',
#     n_jobs=-1,
# )

# start_time = time.time()
# Fit the GridSearchCV object to the training data
# grid_search.fit(X_train, y_train)
# end_time = time.time()
# runtime = end_time - start_time
# print("\n" + "Runtime for GridSearchCV() is: {:.2f} seconds".format(runtime))
# endregion

# # Print the best hyperparameters found
# print("Best hyperparameters: ", grid_search.best_params_)
# print("Best score: ", grid_search.best_score_)

# string_to_save = str(grid_search.best_params_)

# # Specify the file path and name
# file_path = '../../project_BOOK/models_plots/lnAttandance/Best_paramars.txt'
#
# # Open the file in write mode
# # with open(file_path, 'w') as file:
# #     # Write the string to the file
# #     file.write(string_to_save)
#

# # region Evaluate the model with the best hyperparameters on the test data
# best_model = grid_search.best_estimator_
# test_preds = best_model.predict(X_test)
#
# # Compute evaluation metrics
# mae = mean_absolute_error(y_test, test_preds)
# mse = mean_squared_error(y_test, test_preds)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, test_preds)
# endregion

# region XGBoost regression with tuned hyperparameters
# final model
best_model = XGBRegressor(learning_rate=0.05,
                          max_depth=6,
                          n_estimators=200,
                          reg_alpha=0,
                          reg_lambda=10)

# Fit the GridSearchCV object to the training data
best_model.fit(X_train, y_train)

# Save model
joblib.dump(best_model, 'ln_attendance.pkl')

# Evaluate the model with the best hyperparameters on the test data
test_preds = best_model.predict(X_test)

# Fixes the negative prediction
test_preds[test_preds < 0] = 0

# Compute evaluation metrics
mae = mean_absolute_error(y_test, test_preds)
mse = mean_squared_error(y_test, test_preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, test_preds)

# Evaluate the model's feature importancy
for i, column in enumerate(train_data.drop('ln(attendance)', axis=1)):
    # print('Importance of feature {}:, {:.3f}'.format(column, best_xgb_model.feature_importances_[i]))

    fi = pd.DataFrame({'Variable': [column], 'Feature Importance Score': [best_model.feature_importances_[i]]})

    try:
        final_fi = pd.concat([final_fi, fi], ignore_index=True)
    except:
        final_fi = fi

# Ordering the data
final_fi = final_fi.sort_values('Feature Importance Score', ascending=False).reset_index()
print(final_fi)

# Print the evaluation metrics
print("\n" + "Model Score: {:.2f}".format(best_model.score(X_test, y_test)))
print("\n" + "Mean Absolute Error: {:.2f}".format(mae))
print("Mean Squared Error: {:.2f}".format(mse))
print("Root Mean Squared Error: {:.2f}".format(rmse))
print("R-squared: {:.2f}".format(r2))
print()
# endregion

# region learning curve
path = '../../project_BOOK/models_plots/lnAttandance/Learning Curve_BOOK.png'

train_sizes, train_scores, test_scores = learning_curve(best_model, X_train, y_train, cv=5,
                                                        train_sizes=np.linspace(0.1, 1.0, 10), scoring='r2')
# Calculation of the matrics
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()

plt.title("Learning Curve", fontsize=18)
plt.xlabel("Training Examples", fontsize=16)
plt.ylabel("Accuracy Score", fontsize=16)

plt.ylim((0, 1.01))

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                 color='#258B9E')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                 color='#732C91')
plt.plot(train_sizes, train_scores_mean, 'o-', color='#258B9E', label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color='#732C91', label="Cross-validation score")

plt.legend(loc="best")
fig = plt.gcf()
fig.set_size_inches(1920 / 100, 1080 / 100)
plt.savefig(path)
plt.show()
# endregion

# region importance plot
path = '../../project_BOOK/models_plots/lnAttandance/Feature importance_BOOK.png'
plot_importance(best_model, max_num_features=25, color='#258B9E', grid=False, height=0.65)

plt.title("Feature importance", fontsize=18)
plt.xlabel("F Score", fontsize=14)
plt.ylabel("Features", fontsize=14)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

fig = plt.gcf()
fig.set_size_inches(1920 / 100, 1080 / 100)
plt.savefig(path)
plt.show()
# endregion