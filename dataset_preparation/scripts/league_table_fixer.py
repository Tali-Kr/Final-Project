import pandas as pd

lg_tbl = pd.read_csv('../dt_prep_tables/il_league_tables_2012_21.csv')
# lg_tbl = pd.read_csv('../dt_prep_tables/il_league_tables_2022.csv')  # Relevant for only new data

# region lg_tbl addaptation
lg_tbl['game'] = lg_tbl['game'].str.replace('Relegation round', 'Relegation Round')
lg_tbl['game'] = lg_tbl['game'].str.replace('Regular', 'Regular Season')
lg_tbl['goals'] = lg_tbl.apply(lambda x: x['goals'][:-3] if len(x['goals']) == 8 else x['goals'], axis=1)
lg_tbl['season'] = lg_tbl['season'].str.replace('22/23', '2022')
lg_tbl[['goals_for', 'goals_against']] = lg_tbl.goals.str.split(":", expand=True)
lg_tbl['goals_for'] = lg_tbl.apply(lambda x: int(x['goals_for']), axis=1)
lg_tbl['goals_against'] = lg_tbl.apply(lambda x: int(x['goals_against']), axis=1)
lg_tbl.rename(columns={'clubs_name': 'team', 'game': 'round_type'}, inplace=True)
lg_tbl.drop(['goals'], axis='columns', inplace=True)
# endregion

# region league table pts,win,lose,sraw,goals for & against fixed
seasons = lg_tbl['season'].unique()


def last_rnd_rglr_ssn_info():  # Returns the info of the last round of the regular season for each team in each season.
    temp = pd.DataFrame()
    for season in seasons:
        df_fltrd = lg_tbl[(lg_tbl['season'] == season) & (lg_tbl['round_type'] == "Regular Season") &
                          (lg_tbl['round'] == 26)]
        for team in df_fltrd['team'].unique():
            df = df_fltrd[df_fltrd['team'] == team]
            frames = [temp, df]
            temp = pd.concat(frames)
    return temp


lg_tbl_lst_rnd_rglr_ssn = last_rnd_rglr_ssn_info()


def get_info_by_team(team, season, col):  # Returns the info of the team's last round of the regular season.
    return lg_tbl_lst_rnd_rglr_ssn[(lg_tbl_lst_rnd_rglr_ssn['team'] == team) &
                                   (lg_tbl_lst_rnd_rglr_ssn['season'] == season)][col].values[0]


def get_info(record):
    # Fixes the pts, goals_for, goals_against, win, draw, lose info for the championship and relegation rounds,
    # based on the last round of the regular season.
    if record['round_type'] == 'Championship Round' or record['round_type'] == 'Relegation Round':
        cols_list = ['pts', 'goals_for', 'goals_against', 'win', 'draw', 'lose']
        for col in cols_list:
            record[col] = record[col] + get_info_by_team(record['team'], record['season'], col)
        return record
    else:
        return record


lg_tbl = lg_tbl.apply(lambda x: get_info(x), axis=1)
lg_tbl['g_difference'] = lg_tbl.apply(lambda x: (x['goals_for'])-(x['goals_against']), axis=1)
#endregion

#region position fixer
round_types_ = lg_tbl['round_type'].unique()

round_types = {'Regular Season': 26, 'Championship Round': 10, 'Relegation Round': 7}
# round_types = {'Regular Season': 26, 'Championship Round': 7, 'Relegation Round': 6}  # Relevant for only new data
num_team_round_type = {'Regular Season': 14, 'Championship Round': 6, 'Relegation Round': 8}

cols = ['season', 'round', 'round_type', 'team', 'promoted', 'team_pos', 'pts', 'g_difference', 'win', 'draw', 'lose',
        'goals_for', 'goals_against']
lg_tbl = lg_tbl[cols]

# edited for the new data
# lg_tbl['promoted'] = lg_tbl['team'].apply(lambda x: 1 if (x == 'Sekzia Ness Ziona' or
#                                                           x == 'Maccabi Bnei Reineh') else 0)


def position_sort_in_round():
    # Sorts and determines the positions of the teams in the championship and relegation rounds.
    df = pd.DataFrame()
    for season in seasons:
        df_s = pd.DataFrame()
        for r_type in round_types:
            round_num = round_types.get(r_type)
            for i in range(round_num):
                temp = lg_tbl[(lg_tbl['season'] == season) & (lg_tbl['round_type'] == r_type) &
                              (lg_tbl['round'] == i + 1)]
                temp.sort_values(by=['pts', 'g_difference', 'win', 'goals_for'],
                                 inplace=True, ascending=[False, False, False, False])
                temp = temp.reset_index()
                for j in range(num_team_round_type.get(r_type)):
                    temp.at[j, 'team_pos'] = j+1
                frames = [df_s, temp]
                df_s = pd.concat(frames)
        frames = [df, df_s]
        df = pd.concat(frames)
    return df


res = position_sort_in_round()
res = res.reset_index()
res.drop(['level_0', 'index'], axis='columns', inplace=True)
#endregion

res.to_csv('../dt_prep_tables/il_league_tables_2012_21_fixed.csv')
# res.to_csv('../dt_prep_tables/il_league_tables_2022_fixed.csv')  # Relevant for only new data