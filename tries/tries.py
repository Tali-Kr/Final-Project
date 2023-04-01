from champion_relegated_methods import *
import pandas as pd
import math

master = pd.read_csv('master_data_new_updated.csv')

league_t = pd.read_csv('league_tables_test.csv')

key = "Relegation Round - 5 - 2019 - Hapoel Ra'anana"

season = 2019
round_num = 6
game = 'Relegation Round'
team = "Hapoel Ra'anana"

s = game + str(round_num) + " - " + str(season) + ' - ' + team
s_pre = game + str(round_num - 1) + " - " + str(season) + ' - ' + team

pos_6_round_5 = league_t[
    (league_t['game'] == game) & (league_t['season'] == season) & (league_t['round'] == round_num - 1) & (
                league_t['team_pos'] == 6)]
pos_6_round_5 = int(pos_6_round_5['pts'])
team_pos_round_5 = league_t[
    (league_t['game'] == game) & (league_t['season'] == season) & (league_t['round'] == round_num - 1) & (
                league_t['clubs_name'] == team)]
team_pos_round_5 = int(team_pos_round_5['pts'])
differ = pos_6_round_5 - team_pos_round_5


# def test_differ_league_t(df, game, season, round_num, team):
#     pos_6 = df[(df['game'] == game) & (df['season'] == season) & (df['round'] == round_num - 1) & (df['team_pos'] == 6)]
#     pos_6 = int(pos_6['pts'])
#     team_pos = df[
#         (df['game'] == game) & (df['season'] == season) & (df['round'] == round_num - 1) & (df['clubs_name'] == team)]
#     team_pos = int(team_pos['pts'])
#     diff = (pos_6 - team_pos)
#     return diff
#
#
# def test_differ_league_t_first_seacond(df, game, season, round_num, team):
#     pos_1 = df[
#         (df['game'] == game) & (df['season'] == season) & (df['round'] == round_num - 1) & (df['team_pos'] == 1) & (
#                     df['clubs_name'] == team)]
#     pos_1 = int(pos_1['pts'])
#     pos_2 = df[(df['game'] == game) & (df['season'] == season) & (df['round'] == round_num - 1) & (df['team_pos'] == 2)]
#     pos_2 = int(pos_2['pts'])
#     diff = (pos_1 - pos_2)
#     return diff

a = math.isnan(master.at[1,'home'])

print(a)
print(0)
for i in range(len(league_t)):
    if league_t.at[i,'game'] == 'Relegation Round':
        league_t.at[i,'relegated'] = relegation_championship_check(league_t.iloc[i], league_t)
    elif league_t.at[i,'game'] == 'Championship Round':
        league_t.at[i, 'champion'] = relegation_championship_check(league_t.iloc[i], league_t)
    print(i)

print(0)

#     if league_t.at[i, 'game'] == 'Relegation Round':
#         if league_t.at[i, 'round'] > 4:
#             if league_t.at[i, 'team_pos'] > 6:
#                 max_pts = 3 * (7 - (league_t.at[i, 'round'] - 1))
#                 if test_differ_league_t(league_t, league_t.at[i, 'game'], league_t.at[i, 'season'],
#                                         league_t.at[i, 'round'], league_t.at[i, 'clubs_name']) > max_pts:
#                     league_t.at[i, 'relegated'] = 1
#                     print("{} : in the round ** {} ** of season {} the team  ** {} **  played as it was already "
#                           "relegated".format(i,  league_t.at[i, 'round'],league_t.at[i, 'season'],
#                                              league_t.at[i, 'clubs_name']))
#                 else:
#                     league_t.at[i, 'relegated'] = 0
#
#     elif league_t.at[i, 'game'] == 'Championship Round':
#         if league_t.at[i, 'round'] > 6:
#             if league_t.at[i, 'team_pos'] == 1:
#                 max_pts = 3 * (10 - (league_t.at[i, 'round'] - 1))
#                 if test_differ_league_t_first_seacond(league_t, league_t.at[i, 'game'], league_t.at[i, 'season'],
#                                                       league_t.at[i, 'round'], league_t.at[i, 'clubs_name']) > max_pts:
#                     league_t.at[i, 'champion'] = 1
#                     print("{} : in the round ** {} ** of season {} the team  ** {} **  played as it was already "
#                           "CHAMPION".format(i,  league_t.at[i, 'round'],league_t.at[i, 'season'],
#                                                 league_t.at[i, 'clubs_name']))
#                 else:
#                     league_t.at[i, 'champion'] = 0

for i in range(len(master)):
    master.at[i, 'home_relegated'] = league_t[league_t['key'] == master.at[i, 'key_home']]['relegated'].iloc[0]
    master.at[i, 'away_relegated'] = league_t[league_t['key'] == master.at[i, 'key_away']]['relegated'].iloc[0]
    master.at[i, 'home_champion'] = league_t[league_t['key'] == master.at[i, 'key_home']]['champion'].iloc[0]
    master.at[i, 'away_champion'] = league_t[league_t['key'] == master.at[i, 'key_away']]['champion'].iloc[0]

# print (get_record_index_by_team_key(master, key))
# print (test_relegation_pts_by_key(league_t,key))
print(0)
