import math
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from preprocessDf import preprocess
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor, plot_importance
from train_cols import train_cols

# region Functions
def expoRation(att, cap):
    return math.exp(att)/math.exp(cap)
def attBcap(att, cap):
    if att > cap:
        return cap
    else:
        return att
# endregion

# region Import table into DF
df = pd.read_csv('./dt_prep_tables/master_dataset.csv')  # Original final dataset
df_value = pd.read_csv(r'table/final.csv')
df_date_value = pd.read_csv(r'table/clubs_value_by_date.csv')
# endregion

# Preprocess data
df = preprocess(df, df_value, df_date_value)

prediction = int(input('attendance: 0\nratio: 1\n'))
# region What columns will be used for training
if prediction == 0:
    train_cols = ['kot', 'round_type', 'round', 'home_pos_b4_game', 'home_lg_b4_game', 'away_pos_b4_game',
                  'away_lg_b4_game', 'derby', 'while_champion_league', 'while_european_games', 'home_promoted',
                  'home_is_relegated', 'home_is_champion', 'away_promoted', 'away_is_relegated', 'away_is_champion',
                  'capacity', 'ln(attendance)', 'ln(capacity)', 'built_in', 'stadium_age',
                  'stadium_age_squared', 'distance_in_km', 'day_of_week_num'
                  ]
    train_data = df[train_cols].copy(deep=True)
if prediction == 1:
    train_cols = ['kot', 'round_type', 'round', 'home_pos_b4_game', 'home_lg_b4_game', 'away_pos_b4_game',
                  'away_lg_b4_game', 'derby', 'while_champion_league', 'while_european_games', 'home_promoted',
                  'home_is_relegated', 'home_is_champion', 'away_promoted', 'away_is_relegated', 'away_is_champion',
                  'capacity', 'att_ratio', 'ln(capacity)', 'built_in', 'stadium_age',
                  'stadium_age_squared', 'distance_in_km', 'day_of_week_num'
                  ]
    train_data = df[train_cols].copy(deep=True)
# endregion

target = ['ln(attendance)', 'att_ratio']
# region X, y split
X = train_data.drop(target[prediction], axis=1)
y = train_data[target[prediction]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# endregion

# region XGBoost test and tune hyperparameters
# xgb_model = xgb.XGBRegressor()
#
# # Define the parameter grid to search
# param_grid = {
#     'learning_rate': [0.1, 0.01],
#     'max_depth': [3, 5, 7],
#     'n_estimators': [50, 100, 200],
#     'reg_alpha': [0, 0.1, 0.5],
#     'reg_lambda': [0, 0.1, 0.5],
# }
#
# # Create an instance of GridSearchCV to perform the search
# grid_search = GridSearchCV(
#     estimator=xgb_model,
#     param_grid=param_grid,
#     cv=5,
#     scoring='neg_mean_squared_error',
#     n_jobs=-1,
# )
#
# # Fit the GridSearchCV object to the training data
# grid_search.fit(X_train, y_train)
#
# # Print the best hyperparameters found
# print("Best hyperparameters: ", grid_search.best_params_)
#
# # Evaluate the model with the best hyperparameters on the test data
# best_model = grid_search.best_estimator_
# test_preds = best_model.predict(X_test)
#
# # Compute evaluation metrics
# mae = mean_absolute_error(y_test, test_preds)
# mse = mean_squared_error(y_test, test_preds)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, test_preds)
#
# # Print the evaluation metrics
# print("Mean Absolute Error: {:.2f}".format(mae))
# print("Mean Squared Error: {:.2f}".format(mse))
# print("Root Mean Squared Error: {:.2f}".format(rmse))
# print("R-squared: {:.2f}".format(r2))
#
# endregion

# region XGBoost regression with tuned hyperparameters
par = [{'learning_rate': 0.2, 'max_depth': 4, 'min_child_weight': 3, 'gamma': 0.15, 'subsample': 0.95,
        'colsample_bytree': 0.8, 'reg_alpha': 0.02},
       {'learning_rate': 0.2, 'max_depth': 4, 'min_child_weight': 3, 'gamma': 0.15, 'subsample': 0.95,
        'colsample_bytree': 0.8, 'reg_alpha': 0.02}
       ]
best_model = XGBRegressor(learning_rate=par[prediction]['learning_rate'], max_depth=par[prediction]['max_depth'],
                          min_child_weight=par[prediction]['min_child_weight'], gamma=par[prediction]['gamma'],
                          subsample=par[prediction]['subsample'],
                          colsample_bytree=par[prediction]['colsample_bytree'],
                          reg_alpha=par[prediction]['reg_alpha'])

# Fit the GridSearchCV object to the training data
best_model.fit(X_train, y_train)

# Save model
pred_type = ['attendance', 'ratio']
joblib.dump(best_model, pred_type[prediction]+'.pkl')

# Evaluate the model with the best hyperparameters on the test data
test_preds = best_model.predict(X_test)

# Compute evaluation metrics
mae = mean_absolute_error(y_test, test_preds)
mse = mean_squared_error(y_test, test_preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, test_preds)

# Print the evaluation metrics
print("Mean Absolute Error: {:.2f}".format(mae))
print("Mean Squared Error: {:.2f}".format(mse))
print("Root Mean Squared Error: {:.2f}".format(rmse))
print("R-squared: {:.2f}".format(r2))
print()

# plot_importance(best_model)
# plt.show()

# region Create a figure with multiple subplots
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
#
# # Plot the predicted values against the actual values
# axes[0, 0].scatter(y_test, test_preds)
# axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
# axes[0, 0].set_xlabel('Actual')
# axes[0, 0].set_ylabel('Predicted')
# axes[0, 0].set_title('Predicted vs Actual')
#
# # Plot the residuals
# axes[0, 1].scatter(test_preds, y_test - test_preds)
# axes[0, 1].set_xlabel('Predicted')
# axes[0, 1].set_ylabel('Residuals')
# axes[0, 1].set_title('Residuals vs Predicted')
#
# # Plot the histogram of residuals
# axes[1, 0].hist(y_test - test_preds, bins=50)
# axes[1, 0].set_xlabel('Residuals')
# axes[1, 0].set_ylabel('Frequency')
# axes[1, 0].set_title('Distribution of Residuals')
#
# # Plot the evaluation metrics
# axes[1, 1].bar(['MSE', 'MAE', 'RMSE', 'R2'], [mse, mae, rmse, r2])
# axes[1, 1].set_ylabel('Score')
# axes[1, 1].set_title('Evaluation Metrics')
#
# plt.tight_layout()
# plt.show()
# endregion

# endregion
