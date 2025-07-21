# type: ignore

import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
import os

#-----------------------------------------------------------------------
#Takes initial.csv organizes it into an xlsx file with grouped data
#and isolates the signal between trigger 1 and trigger 2.
#The output is saved as xxxx_grouped.xlsx and xxxx_grouped_signal.xlsx
#-----------------------------------------------------------------------

def organize(input_filename):
    wb = openpyxl.Workbook()
    ws = wb.active

    with open(input_filename, "r", encoding="utf-8") as f:
        row_index = 1  
        for line in f:

            line = line.rstrip("\n")
            values = line.split("\t")

            #Put value in next cell in the row
            for col_index, value in enumerate(values, start=1):
                ws.cell(row=row_index, column=col_index, value=value)
            row_index += 1

    
    base_name = os.path.splitext(input_filename)[0] 
    output_filename = base_name + "_grouped.xlsx"

    wb.save(output_filename)
    print(f"Organized file saved as: {output_filename}")
    return output_filename

def signal_isolation(grouped):
    df = pd.read_excel(grouped, engine='openpyxl')

    if 'Trigger' not in df.columns:
        raise ValueError("There is no 'Trigger'.")
    
    trigger1_indices = df.index[df['Trigger'] == 1].tolist()
    trigger2_indices = df.index[df['Trigger'] == 2].tolist()

    if not trigger1_indices or not trigger2_indices:
        raise ValueError("There are no triggers.")
    
    repetitions = []
    i = 0

    for start in trigger1_indices:
        ends = [end for end in trigger2_indices if end > start]
        if ends:
            end = ends[0]
            repetitions.append((start, end))
            trigger2_indices.remove(end)

    if not repetitions:
        print("There were no trigger pairs (1 -> 2)")
        return

    #Save file
    base_name = grouped.replace(".xlsx", "")
    for idx, (start, end) in enumerate(repetitions, start=1):
        df_filtered = df.loc[start:end].copy()
        output_file = f"{base_name}_rep{idx:02d}.xlsx"
        df_filtered.to_excel(output_file, index=False)
        print(f"Saved as file: {output_file}")

if __name__ == "__main__":
    input_file = "fays_thumbs_up.csv"
    grouped = organize(input_file)
    signal_isolation(grouped)

