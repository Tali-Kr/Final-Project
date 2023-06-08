def train_cols(df, prediction):
    # choose the columns that needed to prediction
    # training columns define
    if prediction == 1:
        # Choose training columns home won
        train_cols = ['round', 'kot', 'home_won',
                      'home_pos_b4_game', 'home_lg_b4_game', 'away_pos_b4_game', 'away_lg_b4_game',
                      'kot_18_19', 'kot_19_20', 'kot_20_21', 'while_european_games', 'home_is_relegated',
                      'home_is_champion', 'away_promoted', 'away_is_relegated', 'away_is_champion',
                      'att_ratio', 'ln(attendance)', 'stadium_age', 'distance_in_km', 'day_of_week_num', 'month',
                      'home_value', 'away_value', 'home_underdog', 'away_underdog']
        train_data = df[train_cols].copy(deep=True)
    if prediction == 2:
        # Choose training columns away won
        train_cols = ['round', 'kot', 'away_won', 'home_pos_b4_game', 'home_lg_b4_game', 'away_pos_b4_game',
                      'away_lg_b4_game', 'kot_15_16', 'kot_16_17', 'kot_17_18', 'kot_18_19', 'kot_19_20', 'kot_20_21',
                      'kot_21_22', 'kot_22_23', 'derby', 'while_champion_league', 'while_european_games',
                      'home_promoted', 'home_is_relegated', 'home_is_champion', 'away_promoted', 'away_is_relegated',
                      'away_is_champion', 'att_ratio', 'ln(attendance)', 'stadium_age', 'distance_in_km',
                      'day_of_week_num', 'month', 'home_value', 'away_value', 'home_underdog', 'away_underdog']
        train_data = df[train_cols].copy(deep=True)

    if prediction == 3:
        # Choose training columns for attendance
        train_cols = ['kot', 'season', 'round_type', 'round', 'home_pos_b4_game', 'home_lg_b4_game', 'away_pos_b4_game',
                      'away_lg_b4_game', 'kot_16_17', 'kot_17_18', 'kot_18_19', 'kot_19_20', 'kot_20_21', 'derby',
                      'while_champion_league', 'while_european_games', 'home_promoted', 'home_is_relegated',
                      'home_is_champion', 'away_promoted', 'away_is_relegated', 'away_is_champion', 'ln(attendance)',
                      'ln(capacity)', 'stadium_age', 'distance_in_km', 'day_of_week_num', 'month', 'home_underdog',
                      'away_underdog', 'home_value']
        train_data = df[train_cols].copy(deep=True)

    return train_data
