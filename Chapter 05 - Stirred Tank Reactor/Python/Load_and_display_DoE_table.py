# A script that loads Excel Data File from a Directory and display the DoE Table
# Install the required Python libraries if you haven't already:
# - Pandas and pyDoE3 (for DoE creation and analysis)

import pandas as pd

file_path = 'Mixing_tank_single_response.xlsx'  # If the file is in the current directory
data = pd.read_excel(file_path)   # Load the data from Excel

print('\n========== FULL CCD TABLE ==========')
print(f'Number of runs: {len(data)}')
print(f'Number of variables: {len(data.columns)}\n')
print(data.to_string(index=False))