import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv(r'test.csv')


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


new_df.to_csv('test3.csv')