import json


def get_header_items(items, obj):
    """
    Recursive helper function. Getting the keys of the dictionaries for the header of the table.
    :param items: Empty list that in the end of the function run will contain the headers of the data.
    :param obj: Dictionary from which the headers are taken.
    """
    for x in obj:
        if isinstance(obj[x], dict):  # Check if the object is a key in the dictionary.
            items.append(x)  # If so, the key is appended into the list "items".
            get_header_items(items, obj[x])
        else:
            items.append(x)


def add_items_to_data(items, obj):
    """
    Recursive helper function. Getting the values of the dictionaries for the data of the table.
    :param items: Empty list that in the end of the function run will contain the values of the data.
    :param obj: Dictionary from which the values are taken.
    """
    for x in obj:
        if isinstance(obj[x], dict):  # Check if the object is a key in the dictionary.
            items.append("")  # If so, appending a blank space to the list. The purpose of this line is to "save"
            # space in the list that every value will be under the correct header.
            add_items_to_data(items, obj[x])
        else:
            items.append(obj[x])


def convert_to_csv(file_name):
    """
    Creates CSV file from data of the Json file.
    :param file_name: String. The name of the CSV file that we want to create.
    """
    with open(file_name) as f:  # Opening the json file that we want to convert.
        storing_data = []  # Ampty list that will contain all of the data (keys and values) from the json file.
        data = json.loads(f.read())  # Loading all of the data from the json file to the data variable.
        temp = data[0]  # Inserting the first object of the list to retrieve the headers of the data.
        header_items = []  # Will save the headers of the data
        get_header_items(header_items, temp)
        storing_data.append(header_items)  # Appending the headers of the data.

        for obj in data:  # Loop over the json object and appened the values to the list.
            d = []  # Ampty list that will contain all of the values from the json file.
            add_items_to_data(d, obj)
            storing_data.append(d)  # Appending the values of the data.

        with open(file_name + ".csv", "w", encoding="utf-8") as output_file:  # Puts all the items into CSV file.
            seporater = ","
            for a in storing_data:  # Loop over the items inside the list.
                output = seporater.join(map(str, a)) + "\r"  # Converting into a string. Using "map" so the conversion
                # will happen on every item in the list.
                output_file.write(output)  # Write these items out in the file.


# List of the files name
file_names = ["season_2016", "season_2017", "season_2018", "season_2019", "season_2020", "season_2021"]

# Loop over the list of names to create CSV files
for name in file_names:
    convert_to_csv(name)
