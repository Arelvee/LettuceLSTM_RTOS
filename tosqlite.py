import pandas as pd
import sqlite3

# Load the CSV file
csv_file = 'lettuce_wavelet_cleaned.csv'
df = pd.read_csv(csv_file)

# Connect to SQLite3 (or create one if it doesn't exist)
conn = sqlite3.connect('lettuce_data.db')

# Export the DataFrame to a SQLite table
df.to_sql('lettuce_wavelet', conn, if_exists='replace', index=False)

# Confirm and close
print("CSV has been successfully converted and saved as 'lettuce_data.db' in table 'lettuce_wavelet'.")
conn.close()