import pandas as pd


def train_cols(df, prediction):
    # choose the columns that needed to prediction
    # training columns define
    if prediction == 0:
        # Choose training columns underdog
        train_cols = ['round_type', 'round', 'home_team', 'away_team', 'home_position',
                      'away_position', 'kot_15_16', 'kot_16_17', 'kot_17_18', 'kot_18_19',
                      'kot_19_20', 'kot_20_21', 'kot_21_22', 'kot_22_23', 'derby',
                      'while_world_cup', 'while_champion_league', 'while_european_games',
                      'home_promoted', 'home_league_pts', 'home_is_champion', 'away_promoted',
                      'away_league_pts', 'away_is_champion', 'att_ratio', 'ln(attendance)',
                      'ln(capacity)', 'stadium_age', 'distance_in_km', 'day_of_week_num',
                      'home_value', 'away_value']
        train_data = df[train_cols].copy(deep=True)
    if prediction == 1:
        # Choose training columns home won
        train_cols = ['month', 'round_type', 'round', 'home_pos_b4_game', 'away_pos_b4_game', 'kot', 'derby',
                      'while_champion_league', 'while_european_games', 'home_promoted', 'home_lg_b4_game',
                      'home_is_champion', 'away_promoted', 'away_lg_b4_game', 'away_is_champion', 'att_ratio',
                      'ln(attendance)', 'ln(capacity)', 'stadium_age', 'distance_in_km', 'day_of_week_num',
                      'home_value', 'away_value']
        train_data = df[train_cols].copy(deep=True)
    if prediction == 2:
        # Choose training columns away won
        train_cols = ['round', 'home_pos_b4_game', 'month', 'derby', 'away_pos_b4_game', 'kot', 'while_european_games',
                      'home_promoted', 'away_promoted', 'home_lg_b4_game', 'home_is_champion', 'away_lg_b4_game',
                      'att_ratio', 'ln(attendance)', 'ln(capacity)', 'stadium_age', 'distance_in_km', 'day_of_week_num',
                      'home_value', 'away_value']
        train_data = df[train_cols].copy(deep=True)

    if prediction == 3:
        # Choose training columns for attendance
        train_cols = ['kot', 'round_type', 'round', 'home_pos_b4_game', 'home_lg_b4_game', 'away_pos_b4_game',
                      'away_lg_b4_game', 'derby', 'while_champion_league', 'while_european_games', 'home_promoted',
                      'home_is_relegated', 'home_is_champion', 'away_promoted', 'away_is_relegated', 'away_is_champion',
                      'capacity', 'ln(capacity)', 'built_in', 'stadium_age',
                      'stadium_age_squared', 'distance_in_km', 'day_of_week_num'
                      ]
        train_data = df[train_cols].copy(deep=True)

    if prediction == 4:
        # Choose training columns for ratio
        train_cols = ['kot', 'round_type', 'round', 'home_pos_b4_game', 'home_lg_b4_game', 'away_pos_b4_game',
                      'away_lg_b4_game', 'derby', 'while_champion_league', 'while_european_games', 'home_promoted',
                      'home_is_relegated', 'home_is_champion', 'away_promoted', 'away_is_relegated', 'away_is_champion',
                      'capacity', 'ln(capacity)', 'built_in', 'stadium_age',
                      'stadium_age_squared', 'distance_in_km', 'day_of_week_num'
                      ]
        train_data = df[train_cols].copy(deep=True)

    return train_data
