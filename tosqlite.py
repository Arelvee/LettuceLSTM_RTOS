from datetime import datetime, timedelta
import pandas as pd
import random
import numpy as np
import sqlite3

# Adjusted start date so that 48 days ends on May 31, 2025
end_date = datetime(2025, 5, 31)
start_date = end_date - timedelta(days=48)

# Generate timestamps every 5 minutes for the full range
timestamps = pd.date_range(start=start_date, end=end_date, freq='5T')

# Define the day-based stages
def get_growth_stage(ts):
    days_since_start = (ts - start_date).days + 1
    if days_since_start <= 7:
        return "Seed Sowing"
    elif days_since_start <= 10:
        return "Germination"
    elif days_since_start <= 25:
        return "Leaf Development"
    elif days_since_start <= 44:
        return "Head Formation"
    else:
        return "Harvesting"

# Helper to add noise and occasional outliers
def noisy(value, stddev, outlier_prob=0.005, outlier_range=(-2, 2)):
    noise = np.random.normal(0, stddev)
    value += noise
    if random.random() < outlier_prob:
        value += random.uniform(*outlier_range)
    return round(value, 2)

# Simulate random days with cocopeat influence
cocopeat_days = set(random.sample(range(1, 49), k=10))  # 10 random days affected

# Generate simulated sensor data with noise and realistic modeling
data = []
for ts in timestamps:
    hour = ts.hour
    day = (ts - start_date).days + 1
    stage = get_growth_stage(ts)

    # Humidity and temperatures
    humidity = noisy(random.uniform(40, 70), 2)
    temp_envi = noisy(random.uniform(22, 35), 0.5)
    temp_water = noisy(random.uniform(20, 30), 0.4)

    # Base TDS centered around 630 ppm
    base_tds = 630

    # If cocopeat day, fluctuate TDS
    if day in cocopeat_days:
        tds = noisy(base_tds + random.uniform(-80, 80), 15)
    else:
        tds = noisy(base_tds, 8)

    # Light levels depending on time
    if 5 <= hour < 6 or 18 <= hour or hour < 5:
        # Nighttime LED strip grow light values (lower lux)
        lux = noisy(random.uniform(300, 800), 50)
        ppfd = noisy(random.uniform(100, 250), 20)
    elif 6 <= hour < 12:
        # Morning terrace sun, bright light
        lux = noisy(random.uniform(15000, 25000), 500)
        ppfd = noisy(random.uniform(500, 800), 30)
    elif hour == 12:
        # Midday less direct light due to terrace shading
        lux = noisy(random.uniform(10000, 15000), 400)
        ppfd = noisy(random.uniform(300, 500), 20)
    else:
        # Afternoon dimmer natural light
        lux = noisy(random.uniform(5000, 10000), 400)
        ppfd = noisy(random.uniform(150, 400), 20)

    # Other sensors
    reflect_445 = noisy(random.uniform(0.1, 0.9), 0.03)
    reflect_480 = noisy(random.uniform(0.1, 0.9), 0.03)
    ph = noisy(random.uniform(5.5, 7.5), 0.1, outlier_prob=0.01, outlier_range=(-1, 1))

    data.append([
        ts, humidity, temp_envi, temp_water, tds, lux, ppfd,
        reflect_445, reflect_480, ph, stage
    ])

# Create DataFrame
df = pd.DataFrame(data, columns=[
    "timestamp", "humidity", "temp_envi", "temp_water", "tds",
    "lux", "ppfd", "reflect_445", "reflect_480", "ph", "growth_stage"
])

# Save to CSV
csv_file_path = "/mnt/data/lettuce_growth_finalized.csv"
df.to_csv(csv_file_path, index=False)

# Convert CSV data to SQLite DB
sqlite_db_path = "/mnt/data/lettuce_growth_finalized.db"
conn = sqlite3.connect(sqlite_db_path)
df.to_sql('lettuce_growth', conn, if_exists='replace', index=False)
conn.close()

csv_file_path, sqlite_db_path
