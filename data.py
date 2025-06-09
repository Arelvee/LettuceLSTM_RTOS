import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Start date of planting
start_date = datetime(2025, 4, 9)

# Adjusted growth stages with overlapping
stages = {
    "Seed Sowing": (0, 7),
    "Germination": (6, 10),
    "Seedling Stage": (11, 18),
    "Leaf Development": (19, 32),
    "Head Formation": (33, 38),
    "Harvesting": (39, 45),
}

# Heat surge days (non-linear surges)
heat_surge_dates = {
    datetime(2025, 4, 10).date(),
    datetime(2025, 4, 12).date(),
    datetime(2025, 4, 17).date(),
    datetime(2025, 4, 21).date(),
    datetime(2025, 4, 25).date(),
    datetime(2025, 4, 29).date(),
    datetime(2025, 5, 2).date(),
    datetime(2025, 5, 7).date(),
    datetime(2025, 5, 12).date(),
    datetime(2025, 5, 16).date(),
}

# Parameter ranges per stage
stage_params = {
    "Seed Sowing": {
        "temp_env": (24, 26),
        "temp_water": (24, 25),
        "humidity": (65, 70),
        "ec": (0.9, 1.2),
        "tds": (500, 600),
        "ppfd": (150, 200),
    },
    "Germination": {
        "temp_env": (24, 26),
        "temp_water": (24, 25),
        "humidity": (65, 70),
        "ec": (0.9, 1.2),
        "tds": (500, 600),
        "ppfd": (150, 200),
    },
    "Seedling Stage": {
        "temp_env": (24, 26),
        "temp_water": (24, 25),
        "humidity": (65, 70),
        "ec": (1.0, 1.4),
        "tds": (600, 700),
        "ppfd": (200, 250),
    },
    "Leaf Development": {
        "temp_env": (22, 28),
        "temp_water": (22, 26),
        "humidity": (55, 65),
        "ec": (1.2, 1.5),
        "tds": (700, 800),
        "ppfd": (200, 250),
    },
    "Head Formation": {
        "temp_env": (22, 28),
        "temp_water": (22, 26),
        "humidity": (55, 65),
        "ec": (1.2, 1.5),
        "tds": (700, 800),
        "ppfd": (200, 250),
    },
    "Harvesting": {
        "temp_env": (22, 28),
        "temp_water": (22, 26),
        "humidity": (55, 65),
        "ec": (1.2, 1.5),
        "tds": (700, 800),
        "ppfd": (200, 250),
    },
}

# Function to simulate values
def simulate_param(value_range, surge=False, param=None):
    value = np.random.uniform(*value_range)
    noise = np.random.normal(0, 0.01 * (value_range[1] - value_range[0]))

    # Surge handling
    if surge:
        if param == "temp_env":
            value += random.uniform(2.5, 5.0)
        elif param == "temp_water":
            value += random.uniform(1.0, 2.5)
        elif param == "humidity":
            value -= random.uniform(5.0, 10.0)
    
    # Apply noise and possible missing value
    if random.random() < 0.02:
        return None  # Missing value
    return round(value + noise, 2)

# Final data list
data = []

# Generate data per stage
for stage, (start_day, end_day) in stages.items():
    for day in range(start_day, end_day + 1):
        for hour in range(0, 24):
            timestamp = start_date + timedelta(days=day, hours=hour)
            date = timestamp.date()
            if date > datetime(2025, 5, 16).date():
                break
            params = stage_params[stage]
            surge = date in heat_surge_dates
            ec_value = simulate_param(params["ec"], surge)
            row = {
                "timestamp": timestamp,
                "stage": stage,
                "temp_env": simulate_param(params["temp_env"], surge, "temp_env"),
                "temp_water": simulate_param(params["temp_water"], surge, "temp_water"),
                "humidity": simulate_param(params["humidity"], surge, "humidity"),
                "ec_uScm": ec_value * 1000 if ec_value is not None else None,
                "tds": simulate_param(params["tds"], surge),
                "ppfd": simulate_param(params["ppfd"], surge),
                "reflect_445": round(np.random.uniform(0.05, 0.2), 3) if random.random() > 0.1 else None,
                "reflect_480": round(np.random.uniform(0.05, 0.2), 3) if random.random() > 0.1 else None,
            }
            data.append(row)

# Continue generating data for Harvesting (from May 16 onward)
harvest_start_date = datetime(2025, 5, 16)
harvest_end_date = start_date + timedelta(days=45)  # Day 45 of planting

for day in range(33, 46):  # Harvesting stage is from Day 33 to Day 45
    for hour in range(0, 24):
        timestamp = start_date + timedelta(days=day, hours=hour)
        date = timestamp.date()
        params = stage_params["Harvesting"]
        surge = date in heat_surge_dates
        ec_value = simulate_param(params["ec"], surge)
        row = {
            "timestamp": timestamp,
            "stage": "Harvesting",
            "temp_env": simulate_param(params["temp_env"], surge, "temp_env"),
            "temp_water": simulate_param(params["temp_water"], surge, "temp_water"),
            "humidity": simulate_param(params["humidity"], surge, "humidity"),
            "ec": ec_value * 1000 if ec_value is not None else None,
            "tds": simulate_param(params["tds"], surge),
            "ppfd": simulate_param(params["ppfd"], surge),
            "reflect_445": round(np.random.uniform(0.05, 0.2), 3) if random.random() > 0.1 else None,
            "reflect_480": round(np.random.uniform(0.05, 0.2), 3) if random.random() > 0.1 else None,
        }
        data.append(row)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("plant_growth_data_simulated.csv", index=False)
