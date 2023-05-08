import pandas as pd

lg_tbl = pd.read_csv(r'data_tables/league_tables_2012_2021.csv')

# region lg_tbl addaptation
lg_tbl['game'] = lg_tbl['game'].str.replace('Relegation round', 'Relegation Round')
lg_tbl['game'] = lg_tbl['game'].str.replace('Regular', 'Regular Season')
lg_tbl['goals'] = lg_tbl.apply(lambda x: x['goals'][:-3] if len(x['goals']) == 8 else x['goals'], axis=1)
# region test to see goals length
# lg_tbl['len_goals_clmn'] = lg_tbl.apply(lambda x: len(x['goals']),axis=1)
# a = lg_tbl['len_goals_clmn'].unique()
# b = lg_tbl[lg_tbl['len_goals_clmn'] == 8]
# res = b.apply(lambda x: x['goals'][-2:],axis=1)
# res_u = res.unique()
#
# lg_tbl['goals'] = lg_tbl.apply(lambda x: x['goals'][:-3] if x['len_goals_clmn'] == 8 else x['goals'],axis=1)
# lg_tbl['len_goals_clmn'] = lg_tbl.apply(lambda x: len(x['goals']), axis=1)
# a = lg_tbl['len_goals_clmn'].unique()
# endregion

lg_tbl[['goals_for', 'goals_against']] = lg_tbl.goals.str.split(":", expand=True)
lg_tbl['goals_for'] = lg_tbl.apply(lambda x: int(x['goals_for']), axis=1)
lg_tbl['goals_against'] = lg_tbl.apply(lambda x: int(x['goals_against']), axis=1)

lg_tbl.rename(columns={'clubs_name': 'team', 'game': 'round_type'}, inplace=True)
lg_tbl.drop(['goals'], axis='columns', inplace=True)
# endregion


# region league table pts,win,lose,sraw,goals for & against fixed

seasons = lg_tbl['season'].unique()
def last_rnd_rglr_ssn_info():
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


def get_info_by_team(team, season, col):
    return lg_tbl_lst_rnd_rglr_ssn[(lg_tbl_lst_rnd_rglr_ssn['team'] == team) &
                                   (lg_tbl_lst_rnd_rglr_ssn['season'] == season)][col].values[0]


def get_info(record):
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
num_team_round_type = {'Regular Season': 14, 'Championship Round': 6, 'Relegation Round': 8}

# lg_tbl.drop(['promoted'], axis='columns', inplace=True)

cols = lg_tbl.columns.tolist()

cols = ['season', 'round', 'round_type', 'team', 'promoted', 'team_pos', 'pts', 'g_difference', 'win', 'draw', 'lose',
        'goals_for', 'goals_against']
lg_tbl = lg_tbl[cols]


def position_sort_in_round():
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


res.to_csv('league_tables_2021_2022_fixed.csv')
print()
