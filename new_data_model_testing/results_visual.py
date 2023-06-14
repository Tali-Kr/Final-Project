import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# region Load the data_att from the CSV file
data_att = pd.read_csv('ln_att_predic.csv')
data_home_won = pd.read_csv('home_won_predic.csv')
data_away_won = pd.read_csv('away_won_predic.csv')
df = pd.read_csv('../dataset_preparation/dt_prep_tables/master_dataset.csv')
# endregion

# region best day overlap
# load the csv file
overlap_test = pd.read_csv('../../project_BOOK/models_plots/best_day/all_best_day_lpredictions.csv')

# region attendance Vs away team
# region pi chart
att_away_test = overlap_test[['day_of_week_num', 'day_of_week_num_AWAY']]
att_away_test['same_day'] = att_away_test.apply(lambda x: 1 if x['day_of_week_num'] == x['day_of_week_num_AWAY'] else 0,
                                                axis=1)

# Calculate the counts of each unique value in the "same_day" column
same_day_counts = att_away_test['same_day'].value_counts()

# Define custom colors for the pie chart
colors = ['#258B9E', '#732C91']

# Create a pie chart using the counts and custom colors for inside values
wedges, texts, autotexts = plt.pie(same_day_counts,
                                   autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*same_day_counts.sum())})',
                                   labels=['Different optimal\ndays', 'Same optimal\nday'],
                                   colors=colors, textprops={'color': 'black', 'fontsize': 18})

# Set colors for the outside labels
for autotext in autotexts:
    autotext.set_color('white')

# Draw a white circle in the center to create a donut-like chart
center_circle = plt.Circle((0, 0), 0.40, fc='white')
fig = plt.gcf()
fig.gca().add_artist(center_circle)

# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')
plt.title('Optimal Day for a Game\nAway Team vs. Attendance', fontsize=20)


# save the plot
path = '../../project_BOOK/models_plots/best_day/away_vs_attendance_best_day_BOOK.png'
fig = plt.gcf()
fig.set_size_inches(1920/100, 1080/100)
# plt.savefig(path)

# Display the plot
plt.show()
# endregion

# region day destirbution
# Changing the values from int to representive text
days_dic = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
att_away_test['day_of_week_num'] = att_away_test['day_of_week_num'].apply(lambda x: days_dic.get(x))
att_away_test['day_of_week_num_AWAY'] = att_away_test['day_of_week_num_AWAY'].apply(lambda x: days_dic.get(x))

# Filter the dataframe for games where 'same_day' is equal to 1
filtered_df = att_away_test[att_away_test['same_day'] == 1]

# Count the occurrences of each day of the week
day_counts = filtered_df['day_of_week_num'].value_counts().sort_index()

# Define your custom colors for each day of the week
custom_colors = ['#AEE2EC', '#AEE2EC', '#83D1E1', '#5DC4D9', '#3FB9D1', '#2DA5BD', '#258B9E']
cmap = LinearSegmentedColormap.from_list('custom_colormap', custom_colors)

# Reindex the aggregated values based on the desired order
weekdays = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday']
day_counts = day_counts.reindex(weekdays)

# Define the x and y values for the bar plot
x = day_counts.index
y = day_counts.values

# Create the bar plot
fig, ax = plt.subplots()

# Draw the grid lines
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', zorder=0)  # Set zorder to place the grid lines behind the bars

# Create the bars
bars = ax.bar(x, y, color=custom_colors, zorder=2)  # Set zorder to place the bars in front of the grid lines

# Add the values of the bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height, round(height, 4),
            ha='center', va='bottom', fontsize=16)

# Set the labels and title
ax.set_xlabel('Day of the Week', fontsize=18)
ax.set_ylabel('Predicted Probability', fontsize=18)
ax.set_title('Predicted Probabilities for Away Team to Win by Day of the Week', fontsize=20)

# Adjust the tick font size
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# save the plot
path = '../../project_BOOK/models_plots/best_day/away_vs_attendance_best_day_dister_BOOK.png'
fig = plt.gcf()
fig.set_size_inches(1920/100, 1080/100)
# plt.savefig(path)

# Display the plot
plt.show()
# endregion
# endregion

# region attendance Vs home team
# region pi chart
att_home_test = overlap_test[['day_of_week_num', 'day_of_week_num_HOME']]
att_home_test['same_day'] = att_home_test.apply(lambda x: 1 if x['day_of_week_num'] == x['day_of_week_num_HOME'] else 0,
                                                axis=1)

# Calculate the counts of each unique value in the "same_day" column
same_day_counts = att_home_test['same_day'].value_counts()

# Define custom colors for the pie chart
colors = ['#258B9E', '#732C91']

# Create a pie chart using the counts and custom colors for inside values
wedges, texts, autotexts = plt.pie(same_day_counts,
                                   autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*same_day_counts.sum())})',
                                   labels=['Different optimal\ndays', 'Same optimal\nday'],
                                   colors=colors, textprops={'color': 'black', 'fontsize': 18})

# Set colors for the outside labels
for autotext in autotexts:
    autotext.set_color('white')

# Draw a white circle in the center to create a donut-like chart
center_circle = plt.Circle((0, 0), 0.40, fc='white')
fig = plt.gcf()
fig.gca().add_artist(center_circle)

# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')
plt.title('Optimal Day for a Game\nHome Team vs. Attendance', fontsize=20)


# save the plot
path = '../../project_BOOK/models_plots/best_day/home_vs_attendance_best_day_BOOK.png'
fig = plt.gcf()
fig.set_size_inches(1920/100, 1080/100)
# plt.savefig(path)

# Display the plot
plt.show()
# endregion

# region day destirbution
# Changing the values from int to representive text
days_dic = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
att_home_test['day_of_week_num'] = att_home_test['day_of_week_num'].apply(lambda x: days_dic.get(x))
att_home_test['day_of_week_num_HOME'] = att_home_test['day_of_week_num_HOME'].apply(lambda x: days_dic.get(x))

# Filter the dataframe for games where 'same_day' is equal to 1
filtered_df = att_home_test[att_home_test['same_day'] == 1]

# Count the occurrences of each day of the week
day_counts = filtered_df['day_of_week_num'].value_counts().sort_index()

# Define your custom colors for each day of the week
custom_colors = ['#AEE2EC', '#AEE2EC', '#83D1E1','#5DC4D9', '#3FB9D1', '#2DA5BD','#258B9E']
cmap = LinearSegmentedColormap.from_list('custom_colormap', custom_colors)

# Reindex the aggregated values based on the desired order
weekdays = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday']
day_counts = day_counts.reindex(weekdays)

# Define the x and y values for the bar plot
x = day_counts.index
y = day_counts.values

# Create the bar plot
fig, ax = plt.subplots()

# Draw the grid lines
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', zorder=0)  # Set zorder to place the grid lines behind the bars

# Create the bars
bars = ax.bar(x, y, color=custom_colors, zorder=2)  # Set zorder to place the bars in front of the grid lines

# Add the values of the bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height, round(height, 4),
            ha='center', va='bottom', fontsize=16)

# Set the labels and title
ax.set_xlabel('Day of the Week', fontsize=18)
ax.set_ylabel('Predicted Probability', fontsize=18)
ax.set_title('Predicted Probabilities for Home Team to Win by Day of the Week', fontsize=20)

# Adjust the tick font size
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# save the plot
path = '../../project_BOOK/models_plots/best_day/home_vs_attendance_best_day_dister_BOOK.png'
fig = plt.gcf()
fig.set_size_inches(1920/100, 1080/100)
# plt.savefig(path)

# Display the plot
plt.show()
# endregion

# endregion


#endregion

# region data_away_won
# Load the data
a_best_day = pd.read_csv('../../project_BOOK/models_plots/best_day/away_best_day_153.csv')

# Change that values of the weekday according to the number
days_dic = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
a_best_day['day_of_week'] = a_best_day['day_of_week_num'].apply(lambda x: days_dic.get(x))

# Define your custom colors for each day of the week
custom_colors = ['#258B9E', '#2DA5BD', '#3FB9D1', '#5DC4D9', '#83D1E1', '#AEE2EC', '#AEE2EC']
custom_colors_2 = ['#732C91', '#8633A7', '#A245C7', '#B369D1', '#C286DA', '#D4A9E5', '#D4A9E5']

# Calculate the aggregated values for each day of the week (e.g., mean, median, max)
aggregated_values = a_best_day.groupby('day_of_week')['away_won_pred_prob'].mean()
aggregated_values_win = a_best_day.groupby('day_of_week')['away_won_pred'].mean()

# Reindex the aggregated values based on the desired order
weekdays = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday']
aggregated_values = aggregated_values.reindex(weekdays)
aggregated_values_win = aggregated_values_win.reindex(weekdays)

# Define the x and y values for the bar plot
x = aggregated_values.index
y = aggregated_values.values
y2 = aggregated_values_win.values

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first set of bars on the first subplot
ax1.bar(x, y, color=custom_colors, zorder=2)

ax1.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', zorder=0)  # Set zorder to place the grid lines behind the bars

ax1.set_xlabel('Day of the Week', fontsize=18)
ax1.set_ylabel('Predicted Probability', fontsize=18)
ax1.set_title('Predicted Probabilities for Away Team to\n Win by Day of the Week', fontsize=20)
ax1.set_ylim((0.483, 0.504))

# Plot the second set of bars on the second subplot
ax2.bar(x, y2, color=custom_colors_2, zorder=2)

ax2.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', zorder=0)  # Set zorder to place the grid lines behind the bars

# Set the labels and title
label_size = 12  # Font size for labels

ax2.set_xlabel('Day of the Week', fontsize=18)
ax2.set_ylabel('Predicted Game Result', fontsize=18)
ax2.set_title('Predicted Game Result for \n Away Team by Day of the Week', fontsize=20)

# Set the desired y-axis tick locations
yticks = [0,  1]
ax2.set_yticks(yticks)

# Annotate the values on the bars in Subplot 1
for i, v in enumerate(y):
    ax1.annotate(f'{v:.3f}', xy=(x[i], v), xytext=(0, 5), textcoords='offset points',
                 ha='center', va='bottom', fontsize=16)

# Annotate the values on the bars in Subplot 2
for i, v in enumerate(y2):
    ax2.annotate(f'{v:.0f}', xy=(x[i], v), xytext=(0, 5), textcoords='offset points',
                 ha='center', va='bottom', fontsize=16)

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.2)

# # save the plot
path = '../../project_BOOK/models_plots/best_day/away_best_day_BOOK.png'
fig = plt.gcf()
fig.set_size_inches(1920/100, 1080/100)
# plt.savefig(path)

# Display the plot
plt.show()

# endregion

# region data_home_won
h_best_day = pd.read_csv('../../project_BOOK/models_plots/best_day/home_best_day_167.csv')

days_dic = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
h_best_day['day_of_week'] = h_best_day['day_of_week_num'].apply(lambda x: days_dic.get(x))

# Define your custom colors for each day of the week
custom_colors = ['#258B9E', '#2DA5BD', '#3FB9D1', '#5DC4D9', '#83D1E1', '#AEE2EC', '#AEE2EC']
cmap = LinearSegmentedColormap.from_list('custom_colormap', custom_colors)
# Calculate the aggregated values for each day of the week (e.g., mean, median, max)
aggregated_values = h_best_day.groupby('day_of_week')['home_won_pred_prob'].mean()

# Reindex the aggregated values based on the desired order
weekdays = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday']
aggregated_values = aggregated_values.reindex(weekdays)

# Define the x and y values for the bar plot
x = aggregated_values.index
y = aggregated_values.values

# Create the bar plot
fig, ax = plt.subplots()

# Draw the grid lines
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', zorder=0)  # Set zorder to place the grid lines behind the bars

# Create the bars
bars = ax.bar(x, y, color=custom_colors, zorder=2)  # Set zorder to place the bars in front of the grid lines

# Add the values of the bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height, round(height, 4),
            ha='center', va='bottom', fontsize=16)

# Set the labels and title
ax.set_xlabel('Day of the Week', fontsize=18)
ax.set_ylabel('Predicted Probability', fontsize=18)
ax.set_title('Predicted Probabilities for Home Team to Win by Day of the Week', fontsize=20)
ax.set_ylim((0.692, 0.700))

# Adjust the tick font size
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# save the plot
path = '../../project_BOOK/models_plots/best_day/home_best_day_BOOK.png'
fig = plt.gcf()
fig.set_size_inches(1920/100, 1080/100)
# plt.savefig(path)

# Display the plot
plt.show()

# endregion

# region attendance by day to see best day
df_att = pd.read_csv('../../project_BOOK/models_plots/best_day/att_best_day_hapoelbash_champ_1.csv')

days_dic = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
df_att['day_of_week'] = df_att['day_of_week_num'].apply(lambda x: days_dic.get(x))

# Define the desired order of the x-axis
desired_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday']

att_color = '#2DA5BD'
best_att_color = '#732C91'

# Calculate the average attendance by day
average_attendance = df_att.groupby('day_of_week')['attendance_pred'].mean()

# Reindex the series to match the desired order
average_attendance = average_attendance.reindex(desired_order)

# Create a bar plot
plt.figure(figsize=(8, 6))  # Adjust the figure size as per your preference

# Plotting the bars
bars = plt.bar(average_attendance.index, average_attendance.values, zorder=2, color=att_color)

# Set color for specific days
specific_days = ['Saturday']
for day, bar in zip(average_attendance.index, bars):
    if day in specific_days:
        bar.set_color(best_att_color)
    else:
        bar.set_color(att_color)
    # Add the value of the bar on top
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), int(bar.get_height()),
             ha='center', va='bottom', fontsize=14)

# Adding labels and titles
plt.xlabel('Day of the Week', fontsize=16)
plt.ylabel('Attendance', fontsize=16)
plt.title('Attendance by Day for \n Hapoel Beer-Sheva Vs Hapoel Jerusalem \n 1st Championship round Game ', fontsize=18)

# Add a grid on the y-axis behind the bars
plt.grid(axis='y', linestyle='--', zorder=0)

# Adjust the tick font size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.ylim(8000, 14100)

# save the plot
path = '../../project_BOOK/models_plots/best_day/best_day_att_BOOK.png'
fig = plt.gcf()
fig.set_size_inches(1920/100, 1080/100)
plt.savefig(path)

# Display the plot
plt.show()
# endregion


# region home win plot
# Define color values
color1 = '#D5E8EC'  # White color
color2 = '#207788'  # Desired color

# Define the colormap with two colors
colors = [color1, color2]
cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)

predicted_values = data_home_won['home_won_pred'].values
actual_values = data_home_won['home_won'].values
cf_matrix = confusion_matrix(actual_values, predicted_values)

group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)

# Create the heatmap plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap=cmap, cbar=False, annot_kws={"size": 18}, ax=ax)

plt.title("Confusion Matrix", fontsize=18)

# Set the label size
ax.tick_params(labelsize=14)

# save the plot
path = '../../project_BOOK/models_plots/results/home_predicted_values_BOOK.png'
fig = plt.gcf()
plt.savefig(path)

# Display the plot
plt.show()
# endregion

# region away win plot
predicted_values = data_away_won['away_won_pred'].values
actual_values = data_away_won['away_won'].values
cf_matrix = confusion_matrix(actual_values, predicted_values)

group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)

# Create the heatmap plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap=cmap, cbar=False, annot_kws={"size": 18}, ax=ax)

plt.title("Confusion Matrix", fontsize=18)

# Set the label size
ax.tick_params(labelsize=14)

# save the plot
path = '../../project_BOOK/models_plots/results/away_predicted_values_BOOK.png'
fig = plt.gcf()
plt.savefig(path)

# Display the plot
plt.show()
# endregion

# region attendance scatter plot
# Extract the predicted values and actual target values
ln_predicted_values = data_att['ln(attendance)_pred']
ln_actual_values = data_att['ln(attendance)_orig']

# Calculate the line of perfect fit
max_value = max(ln_predicted_values.max(), ln_actual_values.max())
min_value = min(ln_predicted_values.min(), ln_actual_values.min())
perfect_fit_line = np.linspace(min_value, max_value, 100)

# Create a scatter plot
plt.scatter(ln_predicted_values, ln_actual_values, s=30, c='#258B9E', alpha=0.5)

# Add labels and title
plt.xlabel('Predicted Values', fontsize=16)
plt.ylabel('Actual Target Values', fontsize=16)
plt.title('Prediction Results vs Actual Target Values', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Plot the line of perfect fit
plt.plot(perfect_fit_line, perfect_fit_line, color='#732C91', linestyle='--', label='Line of Perfect Fit', linewidth=1)

# Add legend
plt.legend(loc="best")
path = '../../project_BOOK/models_plots/results/ln_predicted_values_BOOK.png'

# save the plot
fig = plt.gcf()
fig.set_size_inches(1920/100, 1080/100)
# plt.savefig(path)

# Display the plot
plt.show()
# endregion

# region average attendance by day
# Define the desired order of the x-axis
desired_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday']

# Define the colors for specific days and other days
frequent_day_color = '#732C91'
non_frequent_day_color = '#2DA5BD'

# Calculate the average attendance by day
average_attendance = df.groupby('day_of_week')['attendance'].mean()

# Reindex the series to match the desired order
average_attendance = average_attendance.reindex(desired_order)

# Create a bar plot
plt.figure(figsize=(8, 6))  # Adjust the figure size as per your preference

# Plotting the bars
bars = plt.bar(average_attendance.index, average_attendance.values, zorder=2)

# Set color for specific days
specific_days = ['Sunday', 'Monday', 'Saturday']
for day, bar in zip(average_attendance.index, bars):
    if day in specific_days:
        bar.set_color(frequent_day_color)
    else:
        bar.set_color(non_frequent_day_color)

    # Add the value of the bar on top
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), int(bar.get_height()),
             ha='center', va='bottom', fontsize=13)

# Adding labels and titles
plt.xlabel('Day of the Week', fontsize=16)
plt.ylabel('Average Attendance', fontsize=16)
plt.title('Average Attendance by Day (seasons 2012-2021)', fontsize=18)

# Add a grid on the y-axis behind the bars
plt.grid(axis='y', linestyle='--', zorder=0)

# Adjust the tick font size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# save the plot
path = '../../project_BOOK/plots_/Average_Attendance_by_Day.png'
fig = plt.gcf()
fig.set_size_inches(1920/100, 1080/100)
# plt.savefig(path)

# Display the plot
plt.show()
# endregion


print(0)
# region GOOD hearmap att_vs_homeWin_pred
# Create a sequential colormap
cmap = LinearSegmentedColormap.from_list('custom_colormap', ['#2DA5BD', '#732C91'])

days_dic = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
home_best_days = data_home_won['day_of_week_num'].apply(lambda x: days_dic.get(x))
attendance_best_days = data_att['day_of_week_num'].apply(lambda x: days_dic.get(x))

# Create a pivot table to count the overlap between the two columns
overlap_counts = pd.crosstab(attendance_best_days, home_best_days)

# Creating the heatmap
plt.figure(figsize=(8, 6))  # Adjust the figure size as per your preference

sns.heatmap(overlap_counts, cmap=cmap, annot=True, cbar=True, annot_kws={"size": 16})

plt.xlabel('Home Team Won Days', fontsize=14)
plt.ylabel('Attendance Days', fontsize=14)
plt.title('Optimal Days between Attendance Prediction and Home Team Won Prediction', fontsize=18)

x_labels = sorted(home_best_days.unique())[::-1]
y_labels = sorted(attendance_best_days.unique())

plt.xticks(np.arange(len(x_labels)) + 0.5, x_labels, fontsize=12)
plt.yticks(np.arange(len(y_labels)) + 0.5, y_labels, fontsize=12)


# save the plot
path = '../../project_BOOK/models_plots/results/att_vs_homeWin_pred.png'
fig = plt.gcf()
fig.set_size_inches(1920/100, 1080/100)
plt.savefig(path)

# Display the plot
plt.show()
# endregion

att_best_day = pd.read_csv('../../project_BOOK/models_plots/best_day/att_best_day_193.csv')
attttt_best_day = pd.read_csv('../../project_BOOK/models_plots/best_day/dup_new_data.csv')
days_dic = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
attttt_best_day['day_of_week'] = attttt_best_day['day_of_week_num'].apply(lambda x: days_dic.get(x))

# Define your custom colors for each day of the week
custom_colors = ['#258B9E', '#2DA5BD', '#3FB9D1', '#5DC4D9', '#83D1E1', '#AEE2EC', '#AEE2EC']

# Calculate the aggregated values for each day of the week (e.g., mean, median, max)
aggregated_values = attttt_best_day.groupby('day_of_week')['attendance_pred'].mean()

# Reindex the aggregated values based on the desired order
weekdays = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday']
aggregated_values = aggregated_values.reindex(weekdays)

# Define the x and y values for the bar plot
x = aggregated_values.index
y = aggregated_values.values

# Creating the horizontal bar plot
plt.figure(figsize=(8, 6))  # Adjust the figure size as per your preference

plt.barh(x, y, color='blue')

# plt.axvline(x=average_audience, color='red', linestyle='--', label='Average Audience')

plt.xlabel('Audience')
plt.ylabel('Days of the Week')
plt.title('Average Audience by Day of the Week')
plt.xlim(4800, 6000)
plt.legend()

plt.show()
print()









# region best day att


# Assuming you have the predicted audience values and corresponding days in separate lists
predicted_values = data_att['attendance_pred'].values
days_of_week = data_att['day_of_week_num'].values

# Assuming you also have the actual audience values in a separate list
actual_values = data_att['attendance'].values

# Creating a combined visualization
# plt.figure(figsize=(10, 6))  # Adjust the figure size as per your preference
#
# # Scatter plot of predicted values
# plt.scatter(days_of_week, predicted_values, color='blue', label='Predicted')
#
# # Line plot of actual values
# plt.plot(days_of_week, actual_values, color='red', marker='o', linestyle='-', label='Actual')
#
# plt.xlabel('Days of the Week')
# plt.ylabel('Audience')
# plt.title('Predicted vs Actual Audience by Day of the Week')
# plt.legend()
# plt.grid(True)
# plt.xticks(rotation=45)  # Rotating x-axis labels for better readability
#
# plt.show()
print()

# endregion






# region Best day
# region data_att

# # Create a scatter plot for each game
# np.random.seed(42)
#
# plt.scatter(data_att.index, data_att['attendance_pred'], c=data_att['day_of_week_num'], s=100, cmap='Accent', alpha=0.5)
# plt.colorbar(label='Best Day')  # Add colorbar indicating the best day
# plt.xlabel('Game Index')
# plt.ylabel('Number of Audiences')
# plt.title('Best Day Results for Each Game')
# # Add legend
# plt.legend(loc="best")
# path = '../../project_BOOK/models_plots/results/att_best_day.png'
#
# # save the plot
# fig = plt.gcf()
# fig.set_size_inches(1920/100, 1080/100)
# # plt.savefig(path)
#
# # Display the plot
# plt.show()
# # endregion
print()

# # Calculate the count of each best day
# day_counts = data_att['day_of_week_num'].value_counts()
#
# # Create a bar plot
# plt.bar(day_counts.index, day_counts.values)
# plt.xlabel('Day')
# plt.ylabel('Count')
# plt.title('Distribution of Best Days')
# plt.show()















print()

# region data_home_won
h_best_day = pd.read_csv('../../project_BOOK/models_plots/best_day/home_best_day_167.csv')
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame called 'predictions' with the predicted probabilities and the corresponding day of the week
# It should have columns: 'Day of Week' and 'Predicted Probability'

days_dic = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
h_best_day['day_of_week'] = h_best_day['day_of_week_num'].apply(lambda x: days_dic.get(x))

# Define your custom colors for each day of the week
custom_colors = ['#258B9E', '#2DA5BD', '#3FB9D1', '#5DC4D9', '#83D1E1', '#AEE2EC', '#AEE2EC']

# Calculate the aggregated values for each day of the week (e.g., mean, median, max)
aggregated_values = h_best_day.groupby('day_of_week')['home_won_pred_prob'].mean()

# Reindex the aggregated values based on the desired order
weekdays = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday']
aggregated_values = aggregated_values.reindex(weekdays)

# Define the x and y values for the bar plot
x = aggregated_values.index
y = aggregated_values.values

# Customize the bar plot appearance
# color = '#258B9E'  # Custom color for the bars

# Define a colormap with the number of days as the number of colors
# Create an array of colors using the colormap
colors = cmap(np.arange(len(x)))

# Create the bar plot
fig, ax = plt.subplots()

# Draw the grid lines
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', zorder=0)  # Set zorder to place the grid lines behind the bars

# Create the bars
bars = ax.bar(x, y, color=custom_colors, zorder=2)  # Set zorder to place the bars in front of the grid lines

# Add the values of the bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height, round(height, 4),
            ha='center', va='bottom', fontsize=13)

# Set the labels and title
ax.set_xlabel('Day of the Week')
ax.set_ylabel('Predicted Probability')
ax.set_title('Predicted Probabilities for Home Team to Win by Day of the Week')

label_size = 12  # Font size for labels
title = 'Predicted Probabilities for Home Team to Win by Day of the Week'  # Title of the plotSet the labels and title

ax.set_xlabel('Day of the Week', fontsize=label_size)
ax.set_ylabel('Predicted Probability', fontsize=label_size)
ax.set_title(title, fontsize=label_size)
ax.set_ylim((0.692, 0.700))

# save the plot
path = '../../project_BOOK/models_plots/best_day/home_best_d.png'
fig = plt.gcf()
fig.set_size_inches(1920/100, 1080/100)
plt.savefig(path)

# Display the plot
plt.show()
print()
# endregion

# region data_away_won
a_best_day = pd.read_csv('../../project_BOOK/models_plots/best_day/away_best_day_153.csv')
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame called 'predictions' with the predicted probabilities and the corresponding day of the week
# It should have columns: 'Day of Week' and 'Predicted Probability'

days_dic = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
a_best_day['day_of_week'] = a_best_day['day_of_week_num'].apply(lambda x: days_dic.get(x))

# Define your custom colors for each day of the week
custom_colors = ['#258B9E', '#2DA5BD', '#3FB9D1', '#5DC4D9', '#83D1E1', '#AEE2EC', '#AEE2EC']
custom_colors_2 = ['#732C91', '#8633A7', '#A245C7', '#B369D1', '#C286DA', '#D4A9E5', '#D4A9E5']
# Calculate the aggregated values for each day of the week (e.g., mean, median, max)
aggregated_values = a_best_day.groupby('day_of_week')['away_won_pred_prob'].mean()
aggregated_values_win = a_best_day.groupby('day_of_week')['away_won_pred'].mean()

# Reindex the aggregated values based on the desired order
weekdays = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday']
aggregated_values = aggregated_values.reindex(weekdays)
aggregated_values_win = aggregated_values_win.reindex(weekdays)

# Define the x and y values for the bar plot
x = aggregated_values.index
y = aggregated_values.values
y2 = aggregated_values_win.values
# Customize the bar plot appearance
# color = '#258B9E'  # Custom color for the bars

# Define a colormap with the number of days as the number of colors
# Create an array of colors using the colormap
colors = cmap(np.arange(len(x)))

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first set of bars on the first subplot
ax1.bar(x, y, color=custom_colors, zorder=2)

ax1.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', zorder=0)  # Set zorder to place the grid lines behind the bars

# Add the values of the bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, height, round(height, 4),
            ha='center', va='bottom', fontsize=13)

# Set the labels and title
label_size = 12  # Font size for labels
title = 'Predicted Probabilities for Away Team to Win by Day of the Week'  # Title of the plotSet the labels and title

ax1.set_xlabel('Day of the Week', fontsize=label_size)
ax1.set_ylabel('Predicted Probability', fontsize=label_size)
ax1.set_title(title, fontsize=label_size)
ax1.set_ylim((0.483, 0.504))

# Plot the second set of bars on the second subplot
ax2.bar(x, y2, color=custom_colors_2, zorder=2)

ax2.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', zorder=0)  # Set zorder to place the grid lines behind the bars

# Set the labels and title
label_size = 12  # Font size for labels
title = 'Predicted Game Result for Away Team by Day of the Week'  # Title of the plotSet the labels and title

ax2.set_xlabel('Day of the Week', fontsize=label_size)
ax2.set_ylabel('Predicted Game Result', fontsize=label_size)
ax2.set_title(title, fontsize=label_size)

# Set the desired y-axis tick locations
yticks = [0,  1]
ax2.set_yticks(yticks)

# Annotate the values on the bars in Subplot 1
for i, v in enumerate(y):
    ax1.annotate(f'{v:.3f}', xy=(x[i], v), xytext=(0, 5), textcoords='offset points',
                 ha='center', va='bottom')

# Annotate the values on the bars in Subplot 2
for i, v in enumerate(y2):
    ax2.annotate(f'{v:.0f}', xy=(x[i], v), xytext=(0, 5), textcoords='offset points',
                 ha='center', va='bottom')

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.2)

# # save the plot
path = '../../project_BOOK/models_plots/best_day/away_best_d.png'
fig = plt.gcf()
fig.set_size_inches(1920/100, 1080/100)
plt.savefig(path)

# Display the plot
plt.show()
print()
# endregion


# endregion