import pandas as pd
from sklearn.preprocessing import normalize


def preprocess(df, team_value_df):
    # region methods
    def replace_nan(col_names):
        for col_name in col_names:
            df[col_name].fillna(-1, inplace=True)

    def underdog_win(home_pos, away_pos, home_score, away_score):
        if ((home_pos < away_pos) and (home_score < away_score)) or ((home_pos > away_pos) and (home_score > away_score)):
            return 1
        else:
            return 0

    # endregion

    # region Prepare dataframe
    # turn data from yyyy-mm-dd to dd/mm/yyy
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['month'] = df['date'].dt.month
    df['date'] = pd.to_datetime(df['date'], dayfirst=True).dt.strftime('%d/%m/%Y')
    #
    try:
        team_value_df['date'] = pd.to_datetime(team_value_df['date'], dayfirst=True, format='%Y-%m-%d').dt.strftime('%d/%m/%Y')
    except ValueError as a:
        print(a)
    # decide if underdog won or not
    df['udw'] = df.apply(lambda x: underdog_win(x['home_pos_b4_game'], x['away_pos_b4_game'], x['home_score'],
                                                x['away_score']), axis=1)  # could be in the preprocess

    #region decide where the underdog plays
    df['home_underdog'] = df.apply(lambda x: 1 if x['home_team'] == x['underdog'] else 0, axis=1)
    df['away_underdog'] = df.apply(lambda x: 1 if x['away_team'] == x['underdog'] else 0, axis=1)
    #endregion

    team_value_df['unique_key'] = team_value_df['date']+team_value_df['club']
    for col_name in ['home', 'away']:
        df[col_name+'_unique'] = df['date']+df[col_name+'_team']
        df = df.merge(team_value_df[['unique_key', 'value', 'squad_s']], how='left', left_on=col_name+'_unique',
                      right_on='unique_key')
        df.rename(columns={'value': col_name+'_value', 'squad_s': col_name+'_squad_s'}, inplace=True)
        df.drop(['unique_key', col_name+'_unique'], axis=1, inplace=True)

    # region Encode KoT
    # Convert time values to decimal numbers
    df['kot'] = pd.to_datetime(df['kot'], format='%H:%M').dt.hour + pd.to_datetime(df['kot'], format='%H:%M').dt.minute / 60
    # endregion

    # region modifying the round column based on the round_type column
    df.loc[df['round_type'] == 'Relegation Round', 'round'] += 26
    df.loc[df['round_type'] == 'Championship Round', 'round'] += 33
    # endregion

    # region Normalization scaling on value columns **Include in preprocess
    # normalize the team values in each game
    cols_to_normalize = ['home_value', 'away_value']
    # perform normalization scaling on the selected columns
    df[cols_to_normalize] = normalize(df[cols_to_normalize], norm='l2')
    # endregion

    # region encode categorical data
    # encode season and club names
    round_dic = {
        'Regular Season': 0,
        'Relegation Round': 1,
        'Championship Round': 2,
    }
    df['round_type'] = df['round_type'] .apply(lambda x: round_dic.get(x))
    # endregion

    # region add columns to the dataframe for the model
    # # Cumulative wins
    # df['home+season'] = df['home_team']+df['season'].astype(str)
    # df['away+season'] = df['away_team']+df['season'].astype(str)
    #
    # teams = set(df['home+season']) | set(df['away+season'])
    # team_wins = {team: 0 for team in teams}
    #
    # df['home_wins_cumulative'] = 0
    # df['away_wins_cumulative'] = 0
    #
    # for i, row in df.iterrows():
    #     home_team = row['home+season']
    #     away_team = row['away+season']
    #
    #     df.loc[i, 'home_wins_cumulative'] = team_wins[home_team]
    #     df.loc[i, 'away_wins_cumulative'] = team_wins[away_team]
    #
    #     if row['home+season'] > row['away+season']:
    #         team_wins[home_team] += 1
    #     elif row['away_score'] > row['home_score']:
    #         team_wins[away_team] += 1
    # df.drop(['home+season', 'away+season','home_score', 'away_score'], axis=1, inplace=True)
    # endregion
    df.fillna(0, inplace=True)
    return df
    # endregion

