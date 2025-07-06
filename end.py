import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Define optimal ranges for sensors
OPTIMAL_RANGES = {
    'humidity': (65, 75),
    'temp_envi': (22, 35),
    'temp_water': (25, 34),
    'tds': (600, 800),
    'ec': (900, 1600),
    'lux': (15000, 22000),
    'ppfd': (250, 350),
    'reflect_445': (20, 25),
    'reflect_480': (20, 25),
    'ph': (5.8, 6.5)
}

# Create date range: May 31 1am to July 17, every hour
start_date = datetime(2025, 5, 31, 1)
end_date = datetime(2025, 7, 17, 23)
hours = int((end_date - start_date).total_seconds() / 3600) + 1
timestamps = [start_date + timedelta(hours=i) for i in range(hours)]

# Create empty dataframe
df = pd.DataFrame({
    'timestamp': timestamps,
    'batch_id': 1,
    'growth_stage': 'Seed Sowing',
    'yield_count': 0  # Start with 0 yield count
})

# Initialize sensor columns
for sensor in OPTIMAL_RANGES.keys():
    df[sensor] = np.nan

# Historical weather data patterns (simplified)
# Monthly averages: [May, June, July]
monthly_temp_avg = [27.5, 29.0, 30.5]  # Average temperature
monthly_humidity_avg = [70, 75, 72]     # Average humidity

# Water change schedule (every Sunday)
water_change_dates = [
    datetime(2025, 6, 2), datetime(2025, 6, 9), datetime(2025, 6, 16),
    datetime(2025, 6, 23), datetime(2025, 6, 30), datetime(2025, 7, 7),
    datetime(2025, 7, 14)
]

# Growth stage timeline with delay mechanism
base_growth_stages = {
    'Seed Sowing': (datetime(2025, 5, 31), datetime(2025, 6, 6)),
    'Germination': (datetime(2025, 6, 7), datetime(2025, 6, 13)),
    'Leaf Development': (datetime(2025, 6, 14), datetime(2025, 6, 27)),
    'Head Formation': (datetime(2025, 6, 28), datetime(2025, 7, 10)),
    'Harvesting': (datetime(2025, 7, 11), datetime(2025, 7, 17))
}

# Death event simulation (June 9-12) - Caused by neglect
death_start = datetime(2025, 6, 9, 0)
death_end = datetime(2025, 6, 12, 23)
death_duration = (death_end - death_start).days + 1
delayed_stages = {}

# Apply 3-day delay to all stages after Germination
delay_days = 3
for i, (stage, (start, end)) in enumerate(base_growth_stages.items()):
    if stage == 'Seed Sowing' or stage == 'Germination':
        # No delay for these stages
        delayed_stages[stage] = (start, end)
    else:
        # Apply delay to subsequent stages
        new_start = start + timedelta(days=delay_days)
        new_end = end + timedelta(days=delay_days)
        delayed_stages[stage] = (new_start, new_end)

# Adjust Harvesting end date to not exceed July 17
harvest_start, harvest_end = delayed_stages['Harvesting']
if harvest_end > end_date:
    delayed_stages['Harvesting'] = (harvest_start, end_date)

growth_stages = delayed_stages

# Function to get weather-based values
def get_weather_based_values(timestamp):
    """Generate temperature and humidity based on time of day and season"""
    hour = timestamp.hour
    month_idx = timestamp.month - 5  # May=0, June=1, July=2
    
    # Temperature follows daily pattern
    if 0 <= hour < 6:  # Late night/early morning (coolest)
        temp = monthly_temp_avg[month_idx] - 4.5 + random.uniform(-0.2, 0.2)
    elif 6 <= hour < 12:  # Morning (warming)
        temp = monthly_temp_avg[month_idx] - 1.5 + random.uniform(-0.2, 0.2)
    elif 12 <= hour < 18:  # Afternoon (warmest)
        temp = monthly_temp_avg[month_idx] + 2.0 + random.uniform(-0.2, 0.2)
    else:  # Evening (cooling)
        temp = monthly_temp_avg[month_idx] - 1.0 + random.uniform(-0.2, 0.2)
    
    # Humidity inversely related to temperature
    humidity = monthly_humidity_avg[month_idx] + (30 - temp)/2
    humidity += random.uniform(-1, 1)
    
    return temp, humidity

# Function to get light values based on schedule
def get_light_values(timestamp):
    """Generate lux and PPFD values based on time of day"""
    hour = timestamp.hour
    
    if 6 <= hour < 18:  # Outdoor terrace time (6am-6pm)
        # Natural light - peaks at noon
        hour_diff = abs(hour - 12)
        light_factor = max(0.4, 1 - (hour_diff * 0.08))
        lux = 18000 * light_factor + random.uniform(-100, 100)
        ppfd = 300 * light_factor + random.uniform(-5, 5)
    elif 18 <= hour < 22:  # Indoor with grow lights (6pm-10pm)
        # Artificial light - constant from two LED strips
        lux = 19500 + random.uniform(-200, 200)
        ppfd = 320 + random.uniform(-5, 5)
    else:  # Night time (10pm-6am)
        lux = random.uniform(0, 50)  # Minimal ambient light
        ppfd = random.uniform(0, 5)
    
    return lux, ppfd

# Function to get water parameter values
def get_water_parameters(timestamp, last_water_change):
    """Generate TDS, EC, pH with gradual drift between water changes"""
    days_since_change = (timestamp - last_water_change).days
    
    # TDS increases 10-20 ppm per day
    tds = 650 + min(150, days_since_change * 15)
    # EC increases with TDS
    ec = 1000 + min(300, days_since_change * 25)
    # pH decreases slightly over time
    ph = 6.2 - min(0.5, days_since_change * 0.03)
    
    # Add small hourly variation
    tds += random.uniform(-1, 1)
    ec += random.uniform(-2, 2)
    ph += random.uniform(-0.01, 0.01)
    
    return tds, ec, ph

# Function to get reflectance values based on growth stage
def get_reflectance(growth_stage):
    """Generate reflectance values based on growth stage"""
    if growth_stage == 'Seed Sowing':
        return random.uniform(1, 3), random.uniform(1, 3)  # Soil reflectance
    elif growth_stage == 'Germination':
        return random.uniform(4, 8), random.uniform(4, 8)  # Low reflectance (sprouts)
    elif growth_stage == 'Leaf Development':
        return random.uniform(12, 18), random.uniform(12, 18)  # Increasing reflectance
    elif growth_stage == 'Head Formation':
        return random.uniform(20, 25), random.uniform(20, 25)  # Peak reflectance
    elif growth_stage == 'Harvesting':
        return random.uniform(18, 22), random.uniform(18, 22)  # Slightly decreasing
    else:
        return random.uniform(15, 20), random.uniform(15, 20)  # Default

# Generate sensor data
last_water_change = water_change_dates[0]  # Initial water change

for idx, row in df.iterrows():
    timestamp = row['timestamp']
    
    # Determine growth stage based on adjusted timeline
    current_stage = 'Seed Sowing'
    for stage, (start, end) in growth_stages.items():
        if start <= timestamp <= end:
            current_stage = stage
            break
    
    # Update water change if needed
    for wc_date in water_change_dates:
        if timestamp.date() == wc_date.date() and timestamp.hour == 8:  # Morning water change
            last_water_change = wc_date
    
    # Get sensor values
    temp_envi, humidity = get_weather_based_values(timestamp)
    lux, ppfd = get_light_values(timestamp)
    tds, ec, ph = get_water_parameters(timestamp, last_water_change)
    reflect_445, reflect_480 = get_reflectance(current_stage)
    
    # Water temperature follows air temperature with less variation
    temp_water = temp_envi - 1.5 + random.uniform(-0.2, 0.2)
    
    # Set values in dataframe
    df.at[idx, 'temp_envi'] = temp_envi
    df.at[idx, 'humidity'] = humidity
    df.at[idx, 'lux'] = lux
    df.at[idx, 'ppfd'] = ppfd
    df.at[idx, 'tds'] = tds
    df.at[idx, 'ec'] = ec
    df.at[idx, 'ph'] = ph
    df.at[idx, 'temp_water'] = temp_water
    df.at[idx, 'reflect_445'] = reflect_445
    df.at[idx, 'reflect_480'] = reflect_480
    df.at[idx, 'growth_stage'] = current_stage

# Apply death event conditions (June 9-12)
death_mask = (df['timestamp'] >= death_start) & (df['timestamp'] <= death_end)

# Simulate neglect conditions
df.loc[death_mask, 'tds'] = df.loc[death_mask, 'tds'] * 1.25  # TDS too high (750-1000)
df.loc[death_mask, 'ec'] = df.loc[death_mask, 'ec'] * 1.2     # EC too high (1200-1920)
df.loc[death_mask, 'ph'] = df.loc[death_mask, 'ph'] * 0.9     # pH too low (5.2-5.8)
df.loc[death_mask, 'temp_envi'] = df.loc[death_mask, 'temp_envi'] * 1.1  # Temp too high (24-38.5)

# Simulate forgetting to put plants under light during death period
df.loc[death_mask & (df['timestamp'].dt.hour.between(18, 21)), 'lux'] = 100
df.loc[death_mask & (df['timestamp'].dt.hour.between(18, 21)), 'ppfd'] = 15

# Reduce reflectance due to plant stress
df.loc[death_mask, 'reflect_445'] = df.loc[death_mask, 'reflect_445'] * 0.7
df.loc[death_mask, 'reflect_480'] = df.loc[death_mask, 'reflect_480'] * 0.7

# YIELD COUNT ADJUSTMENT (0-6 scale with only 2 surviving plants)
# --------------------------------------------------------------
# Reset yield_count to 0 for all rows
df['yield_count'] = 0

# Set initial plants to 6 (healthy at start)
initial_mask = (df['timestamp'] >= start_date) & (df['timestamp'] < death_start)
df.loc[initial_mask, 'yield_count'] = 6

# Gradually decrease plants during death event (June 9-12)
death_dates = [
    datetime(2025, 6, 9),
    datetime(2025, 6, 10),
    datetime(2025, 6, 11),
    datetime(2025, 6, 12)
]

# Daily plant count during death event
death_counts = [5, 4, 3, 2]

for date, count in zip(death_dates, death_counts):
    mask = (df['timestamp'].dt.date == date.date())
    df.loc[mask, 'yield_count'] = count

# After death event, only 2 plants survived
post_death_mask = (df['timestamp'] > death_end)
df.loc[post_death_mask, 'yield_count'] = 2

# Ensure values stay within physical limits
df['humidity'] = df['humidity'].clip(40, 90)
df['temp_envi'] = df['temp_envi'].clip(18, 40)
df['temp_water'] = df['temp_water'].clip(20, 36)
df['ph'] = df['ph'].clip(5.0, 7.0)
df['reflect_445'] = df['reflect_445'].clip(0, 100)
df['reflect_480'] = df['reflect_480'].clip(0, 100)
df['lux'] = df['lux'].clip(0, 30000)
df['ppfd'] = df['ppfd'].clip(0, 500)
df['tds'] = df['tds'].clip(400, 1200)
df['ec'] = df['ec'].clip(600, 2000)

# Save to CSV
df.to_csv("sensor_growth_data.csv", index=False)

# Print verification
print("Adjusted Growth Stage Timeline:")
for stage, (start, end) in growth_stages.items():
    count = df[df['growth_stage'] == stage].shape[0]
    print(f"- {stage}: {start.date()} to {end.date()} ({count} records)")

print("\nYield Count Summary:")
print(f"- Initial plants: 6")
print(f"- During death event: Gradually decreased from 5 to 2")
print(f"- After death event: Only 2 plants survived")
print(f"- Final yield count: {df[df['timestamp'] == end_date]['yield_count'].values[0]}")

print("\nDataset generated successfully with:")
print(f"- Realistic sensor patterns")
print(f"- 3-day growth delay after death event")
print(f"- Death event from {death_start.date()} to {death_end.date()}")
print(f"- Only 2 surviving lettuce plants")
print(f"Data shape: {df.shape}, From {df['timestamp'].min()} to {df['timestamp'].max()}")