# Distance between two stadiums
master_df['distance_in_km'] = master_df.apply(
    lambda x: geopy.distance.geodesic(x['home_coordinates'], x['away_coordinates']).km, axis=1)