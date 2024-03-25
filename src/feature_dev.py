import pandas as pd
import os
import random

# Change the directory to 'dataset'
os.chdir('dataset')
# Path to the CSV file
csv_file = 'Truck_and_Bus_Through_Route.csv'

# Read the CSV file into a dataframe
df = pd.read_csv(csv_file)

# define feature TRUCK_BREAK_OFF
df['TRUCK_BREAK_OFF'] = 0

# Randomize 0s and 1s for the column TRUCK_BREAK_OFF
df['TRUCK_BREAK_OFF'] = [random.randint(0, 1) for _ in range(len(df))]

print(df.head(5))