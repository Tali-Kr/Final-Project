def while_world_cup(d):
    """
    Checks if the match took place while the world cups were on (in 2014 and 2018).
    :param d: date. Game's date that is checked if took place while one of the World Cups.
    """
    # List of the start dates of the World Cups of 2014 and 2018.
    world_cup_start = list(map(lambda x: datetime.strptime(x, '%d-%m-%Y').date(), ['12-06-2014', '14-06-2018']))
    # List of the end dates of the World Cups of 2014 and 2018.
    world_cup_end = list(map(lambda x: datetime.strptime(x, '%d-%m-%Y').date(), ['13-07-2014', '15-07-2018']))
    return sum(list(map(lambda x, y: 1 if x < d < y else 0, world_cup_start, world_cup_end)))