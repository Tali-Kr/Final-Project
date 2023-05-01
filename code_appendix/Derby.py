def get_cities(team):
    """
    Extracts the city's name from the club's name.
    All of the clubs have their home city in the club's name.
    :param team: string. Team's name that needed to get the city's name.
    :return: string. City's name
    """
    club_city_name = team.split(' ', 1)  # Splits the team's name to seperate the city's name.
    if len(club_city_name) == 1:  # Addressing the option that a club may have only a one-word name.
        return str(club_city_name)
    else:
        return str(club_city_name[1])


def is_derby(record):
    """
    Checks if the game is a derby. Meaning the function checks if the home and away teams are from the same city.
    :param record: Series. A row from the dataframe that needs to check if the game is a derby.
    :return: 1 => if the game is a derby game.
             0 => if the game is NOT a derby game.
    """
    if get_cities(record['home_team']) == get_cities(record['away_team']):  # Checks the game if a city's derby.
        return 1
    else:  # The game isn't a derby game.
        return 0