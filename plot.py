# type: ignore

import pandas as pd
import matplotlib.pyplot as plt

#-----------------------------------------------
#Takes xxxx_grouped.xlsx and plots the channels
#-----------------------------------------------

file_path = 'alikas_thumbs_up_grouped.xlsx'  
df = pd.read_excel(file_path, engine='openpyxl')

timestamp_column = df.columns[-1]

# UNIX timestamp σε datetime format
df[timestamp_column] = pd.to_datetime(df[timestamp_column], unit='s')

#channels = df.columns[:8]  
channels = ['Channel1']

plt.figure(figsize=(15, 7))
for channel in channels:
    plt.plot(df[timestamp_column], df[channel], label=channel) 


plt.xlabel('Time')
plt.ylabel('Voltage')
plt.title('Plot of All Channels')
plt.legend()
plt.grid(True)

plt.show()
