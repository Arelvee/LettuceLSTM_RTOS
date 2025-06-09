import sqlite3
import pandas as pd

# Step 1: Connect to your SQLite database
conn = sqlite3.connect('detectionTwo.db')  # Change this to your actual DB file

# Step 2: Get all table names
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Step 3: Create a Pandas Excel writer
with pd.ExcelWriter('output.xlsx', engine='openpyxl') as writer:
    for table_name in tables:
        table = table_name[0]
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        df.to_excel(writer, sheet_name=table, index=False)

# Step 4: Close the connection
conn.close()