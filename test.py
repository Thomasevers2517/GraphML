import var
import os 
import sys
import pandas as pd

dataset = var.load_data()
print(dataset[0].head())
all_data = pd.concat(dataset)
print(all_data.head(100))

dates = all_data['date'].unique().tolist()
print(dates)

frame_size = all_data.shape
print("Size of the DataFrame:", frame_size)