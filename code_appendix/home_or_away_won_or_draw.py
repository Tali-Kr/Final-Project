# Determining which team won or if the game ended with a draw.
master_df['home_won'] = master_df.apply(lambda x: 1 if x['home_score'] > x['away_score'] else 0, axis=1)
master_df['away_won'] = master_df.apply(lambda x: 1 if x['home_score'] < x['away_score'] else 0, axis=1)
master_df['draw'] = master_df.apply(lambda x: 1 if x['home_score'] == x['away_score'] else 0, axis=1)