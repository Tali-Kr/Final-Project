import pandas as pd
import numpy as np
import datetime
import geopy.distance

def division(num, den):
    num = float(str(num).replace(",", ""))
    den = float(str(den).replace(",", ""))
    return num/den

pd.set_option("display.max_columns", None)

# Import tables and turn them to a dataframe
ligat_haal = pd.read_csv(r'tables\ligat_haal.csv')
stadiums_in_israel = pd.read_csv(r'tables\Stadiums_In_Israel.csv')
club_home_stadium = pd.read_csv(r'tables\club_home_stadium.csv')
stadium_match = pd.read_csv(r'tables\stadium_match.csv')
ds = pd.read_csv(r'tables\master_data.csv')

# Removing unnecessary columns
ds.drop(['Unnamed: 0', 'fixture', 'referee', 'timezone', 'timestamp', 'periods', 'first', 'second', 'venue',
         'venueID', 'city', 'status', 'long', 'short', 'elapsed', 'league', 'leagueID', 'leagueName', 'country', 'logo',
         'flag', 'season', 'round', 'teams', 'home', 'homeTeamID', 'homeTeamLogo', 'homeWinner', 'away', 'awayTeamID',
         'awayTeamLogo', 'awayWinner', 'goals', 'score', 'halftime', 'homeHalfTimeScore', 'awayhomeHalfTimeScore',
         'fulltime', 'homeFullTimeScore', 'awayFullTimeScore', 'extratime', 'homeExtraTimeScore', 'awayExtraTimeScore',
         'penalty', 'homePenaltyTimeScore', 'awayPenaltyTimeScore'], axis='columns', inplace=True)

# fix date
ds['date'] = ds['date'].str[8:10]+"/"+ds['date'].str[5:7]+"/"+ds['date'].str[:4]

# Remove all the gibberish from the stadium name
temp = stadium_match['stadium_name_in_ds'].str.split(pat="(", n=1, expand=True)
stadium_match['name'] = temp[0].str.strip()
stadium_match.drop(['stadium_name_in_ds'], axis='columns', inplace=True)

# Remove all the gibberish from the stadium name
temp = ds['name'].str.split(pat="(", n=1, expand=True)
ds.drop(['name'], axis='columns', inplace=True)
ds['name'] = temp[0].str.strip()

# join two tables
df = pd.merge(ds, stadium_match, how='left', left_on=['name'], right_on=['name'])
df.drop_duplicates(subset=['gameID'], keep='last', inplace=True)
df.drop(['name'], axis='columns', inplace=True)
df.rename(columns={'stadium_name_in_stadium_ds': 'home_team_stadium'}, inplace=True)

# Change the name "Hapoel Katamon" to "Hapoel Katamon Jerusalem"
df['homeTeamName'] = df['homeTeamName'].str.replace("Hapoel Katamon", "Hapoel Katamon Jerusalem")
df['awayTeamName'] = df['awayTeamName'].str.replace("Hapoel Katamon", "Hapoel Katamon Jerusalem")

# # add home team stadium
# df = pd.merge(df, club_home_stadium, how='left', left_on=['homeTeamName'], right_on=['club'])
# df.rename(columns={'home_stadium': 'home_team_stadium'}, inplace=True)
# df.drop(['club'], axis='columns', inplace=True)

# Add away team stadium
df = pd.merge(df, club_home_stadium, how='left', left_on=['awayTeamName'], right_on=['club'])
df.rename(columns={'home_stadium': 'away_team_stadium'}, inplace=True)
df.drop(['club'], axis='columns', inplace=True)

# Game Attendance
# Create Unique Key to merge two tables
df['unique_key1'] = df['date']+df['homeTeamName']+df['awayTeamName']
ligat_haal['unique_key1'] = ligat_haal['match_date']+ligat_haal['home_team']+ligat_haal['away_team']

#Join two tables for the attendance
df = pd.merge(df, ligat_haal[['unique_key1', 'attendance']], how='left', left_on=['unique_key1'], right_on=['unique_key1'])
df.drop(['unique_key1'], axis='columns', inplace=True)

# Get stadium capacity
df = pd.merge(df, stadiums_in_israel[['Stadium', 'Capacity']], how='left', left_on=['home_team_stadium'], right_on=['Stadium'])
df.drop(['Stadium'], axis='columns', inplace=True)
df.rename(columns={'Capacity': 'home_stadium_capacity'}, inplace=True)

# attendance ratio
# df['att_ratio'] = df['attendance'].convert_dtypes(convert_integer=True)/df['home_stadium_capacity'].convert_dtypes(convert_integer=True)
df['att_ratio'] = df.apply(lambda x: division(x['attendance'], x['home_stadium_capacity']), axis=1)

# Get stadium build year
df = pd.merge(df, stadiums_in_israel[['Stadium', 'built_in']], how='left', left_on=['home_team_stadium'], right_on=['Stadium'])
df.drop(['Stadium'], axis='columns', inplace=True)

# Get stadium age
df['stadium_age'] = datetime.date.today().year - df['built_in'].convert_dtypes(convert_integer=True)

# Get coordinates of home team
df = pd.merge(df, stadiums_in_israel[['Stadium', 'coordinates']], how='left', left_on=['home_team_stadium'], right_on=['Stadium'])
df.drop(['Stadium'], axis='columns', inplace=True)
df.rename(columns={'coordinates': 'home_team_stadium_coordinates'}, inplace=True)

# Get coordinates of away team stadium
df = pd.merge(df, stadiums_in_israel[['Stadium', 'coordinates']], how='left', left_on=['away_team_stadium'], right_on=['Stadium'])
df.drop(['Stadium'], axis='columns', inplace=True)
df.rename(columns={'coordinates': 'away_team_stadium_coordinates'}, inplace=True)

# Distance between two stadiums
df['distance_in_km'] = df.apply(lambda x: geopy.distance.geodesic(x['home_team_stadium_coordinates'], x['away_team_stadium_coordinates']).km, axis=1)
df.drop(['home_team_stadium_coordinates', 'away_team_stadium_coordinates'], axis='columns', inplace=True)

df.to_csv('test.csv')
