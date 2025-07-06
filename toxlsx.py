import sqlite3
import pandas as pd

# --- Step 1: Connect to your SQLite database
conn = sqlite3.connect('prediction_data.db')  # change to your database file

# --- Step 2: List all tables (optional, helpful for large DBs)
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("Tables in DB:\n", tables)

# --- Step 3: Choose a table and load it into a DataFrame
table_name = 'forecast_batches'  # change this to your actual table name
df = pd.read_sql(f"SELECT * FROM {table_name}", conn)

# --- Step 4: Save the DataFrame to an Excel file
df.to_excel('output.xlsx', index=False)  # saves to Excel without row index

# --- Step 5: Close the connection
conn.close()
