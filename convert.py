import sqlite3
import pandas as pd

# Read your generated CSV data
df = pd.read_csv("sensor_growth_data.csv")

# Convert timestamp string to datetime object
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Connect to SQLite database (creates new file)
conn = sqlite3.connect('hydroponic_growth_data.db')

# Export DataFrame to SQLite
df.to_sql(
    name='lettuce_wavelet',         # Table name
    con=conn,                   # Database connection
    if_exists='replace',        # Replace table if exists
    index=False,                # Don't include DataFrame index
    dtype={                     # Specify datatypes for columns
        'timestamp': 'DATETIME',
        'batch_id': 'INTEGER',
        'growth_stage': 'TEXT',
        'yield_count': 'INTEGER',
        'humidity': 'REAL',
        'temp_envi': 'REAL',
        'temp_water': 'REAL',
        'tds': 'REAL',
        'ec': 'REAL',
        'lux': 'REAL',
        'ppfd': 'REAL',
        'reflect_445': 'REAL',
        'reflect_480': 'REAL',
        'ph': 'REAL'
    }
)

# Close database connection
conn.close()

print("Data successfully exported to SQLite database: hydroponic_growth_data.db")
print(f"Table created: 'lettuce_wavelet' with {len(df)} records")