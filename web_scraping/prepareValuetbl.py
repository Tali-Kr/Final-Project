import pandas as pd
import numpy as np

# import scraped table and change 0 to nan
df = pd.read_csv(r'temp_team_values.csv')
df = df.replace(0, np.nan)
df.drop(['Unnamed: 0'], axis=1, inplace=True)

# fill the missing values with the last value of the same club the rest that doesn't exist fill with mean
for name in ['value', 'squad_s']:
    df[name] = df.groupby(['club'])[name].ffill()
    df[name] = df.groupby("club")[name].transform(lambda x: x.fillna(x.mean()))

# convert date column to datetime type
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

# create new dataframe with rows for each date
new_df = pd.DataFrame(columns=df.columns)
for club in df['club'].unique():
    club_df = df[df['club'] == club].copy()
    dates = pd.date_range(start=club_df['date'].min(), end=club_df['date'].max())
    club_df = club_df.set_index('date').reindex(dates).fillna(method='ffill').reset_index()
    club_df['club'] = club
    new_df = pd.concat([new_df, club_df], ignore_index=True)
# fix the columns
new_df.drop(['date'], axis=1, inplace=True)
new_df.rename(columns={'index': 'date'}, inplace=True)
new_df = new_df[['date', 'club', 'value', 'squad_s']]

# Save the file
new_df.to_csv('teams_values.csv', index=False)