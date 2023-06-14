import pandas as pd
import joblib
from new_data_model_testing.max_predictions import get_best_day

# what do you want to check:
prediction = int(input('For what do you want to know the best day?\n'
                       'Optimal match day for the audience: 0\n'
                       'Optimal day for the home team to win: 1\n'
                       'Optimal day for the away team to win: 2\n'))


