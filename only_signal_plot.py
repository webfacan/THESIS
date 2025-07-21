# type: ignore

import pandas as pd
import matplotlib.pyplot as plt

#-------------------------------------------------------
#Takes xxxx_grouped_signal.xlsx and plots the channels
#-------------------------------------------------------

input_file = 'alikas_thumbs_up_grouped_signal.xlsx'
df = pd.read_excel(input_file, engine='openpyxl')

#channel_columns = ['Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6', 'Channel7', 'Channel8']  # Προσαρμόστε αν υπάρχουν περισσότερα
channel_columns = ['Channel1']
timestamp_column = df.columns[-1]  
df[timestamp_column] = pd.to_datetime(df[timestamp_column], unit='s')

plt.figure(figsize=(15, 7))
for channel in channel_columns:
    if channel in df.columns: 
        plt.plot(df[timestamp_column], df[channel], label=channel)

plt.xlabel('Time')
plt.ylabel('Voltage')
plt.title('Plot of Signal')
plt.legend()
plt.grid(True)

plt.show()
