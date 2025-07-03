from datetime import datetime, timedelta
import pandas as pd
import random
import numpy as np

# --- 1. Date Configuration ---
end_date = datetime(2025, 5, 31)
start_date = end_date - timedelta(days=48)
timestamps = pd.date_range(start=start_date, end=end_date, freq='5min')

# --- 2. Growth Stage Mapping ---
def get_growth_stage(ts):
    days_since_start = (ts - start_date).days + 1
    if days_since_start <= 8:
        return "Seed Sowing"
    elif days_since_start <= 11:
        return "Germination"
    elif days_since_start <= 25:
        return "Leaf Development"
    elif days_since_start <= 44:
        return "Head Formation"
    else:
        return "Harvesting"

# --- 3. Noise Function with Optional Outliers ---
def noisy(value, stddev, outlier_prob=0.005, outlier_range=(-2, 2)):
    noise = np.random.normal(0, stddev)
    value += noise
    if random.random() < outlier_prob:
        value += random.uniform(*outlier_range)
    return round(value, 2)

# --- 4. Cocopeat Influence Setup (Random 10 Days) ---
cocopeat_days = set(random.sample(range(1, 49), k=10))  # 10 random affected days

# --- 5. Data Generation ---
data = []
batch_id = "Batch_001"

for ts in timestamps:
    hour = ts.hour
    day = (ts - start_date).days + 1
    stage = get_growth_stage(ts)

    # Humidity
    if 0 <= hour < 4:
        humidity = noisy(random.uniform(80, 85), 2)
    elif 4 <= hour < 12:
        humidity = noisy(random.uniform(58, 60), 2)
    elif 12 <= hour < 16:
        humidity = noisy(random.uniform(44, 54), 2)
    else:
        humidity = noisy(random.uniform(60, 70), 2)

    # Environmental Temperature
    if 0 <= hour < 6:
        temp_envi = noisy(random.uniform(22, 26), 0.3)
    elif 6 <= hour < 12:
        temp_envi = noisy(random.uniform(26, 32), 0.4)
    elif 12 <= hour < 14:
        temp_envi = noisy(random.uniform(34, 37), 0.3)
    elif 14 <= hour < 16:
        temp_envi = noisy(random.uniform(37, 38), 0.3)
    elif 16 <= hour < 18:
        temp_envi = noisy(random.uniform(30, 34), 0.3)
    else:
        temp_envi = noisy(random.uniform(24, 30), 0.3)

    # Water Temperature
    if 0 <= hour < 6:
        temp_water = noisy(random.uniform(20, 24), 0.2)
    elif 6 <= hour < 12:
        temp_water = noisy(random.uniform(24, 28), 0.3)
    elif 12 <= hour < 16:
        temp_water = noisy(random.uniform(32, 34), 0.2)
    else:
        temp_water = noisy(random.uniform(26, 30), 0.2)

    # TDS
    base_tds = 630
    if day in cocopeat_days:
        tds = noisy(base_tds + random.uniform(-80, 80), 10)
    else:
        tds = noisy(base_tds, 6)

    # EC (mS/cm) based on TDS
    ec = round(tds / 640, 2)  # Common hydroponic conversion (1 EC ≈ 640 ppm TDS)

    # Light and PPFD
    if 5 <= hour < 6 or 18 <= hour or hour < 5:
        lux = noisy(random.uniform(500, 1000), 30)
        ppfd = noisy(random.uniform(150, 250), 10)
    elif 6 <= hour < 12:
        lux = noisy(random.uniform(15000, 20000), 300)
        ppfd = noisy(random.uniform(500, 700), 20)
    elif hour == 12:
        lux = noisy(random.uniform(10000, 14000), 250)
        ppfd = noisy(random.uniform(300, 500), 15)
    else:
        lux = noisy(random.uniform(5000, 8000), 200)
        ppfd = noisy(random.uniform(200, 400), 15)

    # Reflectance values based on growth stage
    reflectance_ranges = {
        "Seed Sowing": (0.20, 0.40, 0.15, 0.35),
        "Germination": (0.15, 0.30, 0.12, 0.28),
        "Leaf Development": (0.08, 0.20, 0.07, 0.18),
        "Head Formation": (0.05, 0.15, 0.04, 0.12),
        "Harvesting": (0.10, 0.25, 0.08, 0.20),
    }
    r445_min, r445_max, r480_min, r480_max = reflectance_ranges[stage]
    reflect_445 = noisy(random.uniform(r445_min, r445_max), 0.01)
    reflect_480 = noisy(random.uniform(r480_min, r480_max), 0.01)

    # pH (base) with gradual rise if unattended (simulate overtime drift)
    if day < 30:
        ph = noisy(random.uniform(5.8, 6.2), 0.05, outlier_prob=0.005, outlier_range=(0.2, 0.5))
    elif 15 <= day < 40:
        ph = noisy(random.uniform(6.2, 6.5), 0.05, outlier_prob=0.005, outlier_range=(0.2, 0.4))
    else:
        ph = noisy(random.uniform(6.5, 6.8), 0.05, outlier_prob=0.01, outlier_range=(0.2, 0.6))

    # Yield Count (plant growth tracking)
    if stage == "Harvesting":
        yield_prediction = 6
    elif stage == "Germination":
        if day == 9:
            yield_prediction = 1
        elif day == 10:
            yield_prediction = 3
        elif day == 11:
            yield_prediction = 5
        else:
            yield_prediction = 6
    elif day >= 12:
        yield_prediction = 6
    else:
        yield_prediction = 0

    # Append record
    data.append([
        batch_id, ts, day, humidity, temp_envi, temp_water, tds, ec, lux, ppfd,
        reflect_445, reflect_480, ph, stage, yield_prediction
    ])

# --- 6. Create DataFrame (no file saving in this notebook) ---
df = pd.DataFrame(data, columns=[
    "batch_id", "timestamp", "day_number", "humidity", "temp_envi", "temp_water", "tds", "ec",
    "lux", "ppfd", "reflect_445", "reflect_480", "ph", "growth_stage", "yield_count"
])


# Save the generated DataFrame to CSV
file_path = "lettuce_growth_with_reflectance_ec_ph.csv"
df.to_csv(file_path, index=False)

file_path  # Provide path for download

