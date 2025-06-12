import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model, Model
from xgboost import XGBRegressor, XGBClassifier
import pickle

# -----------------------------
# 🧠 Load Models
# -----------------------------
lstm_model = load_model("saved_models/lstm_feature_extractor.keras")
xgb_reg = XGBRegressor()
xgb_reg.load_model("saved_models/xgb_reg.json")
xgb_clf = XGBClassifier()
xgb_clf.load_model("saved_models/xgb_clf.json")

with open("saved_models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
with open("saved_models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# ⚙️ Config
# -----------------------------
seq_len = lstm_model.input_shape[1]
FEATURE_COLS = ['humidity', 'temp_envi', 'temp_water', 'tds', 'lux', 'ppfd', 'reflect_445', 'reflect_480', 'ph']
START_DATE = datetime(2025, 5, 31)
END_DATE = datetime(2025, 8, 29)
BATCH_DURATION = timedelta(days=46)
CURRENT_DATE = datetime(2025, 6, 12)

# -----------------------------
# 📥 Load Sensor Data
# -----------------------------
conn = sqlite3.connect("lettuce_data.db")
df = pd.read_sql_query("SELECT * FROM lettuce_wavelet ORDER BY timestamp ASC", conn)
conn.close()
df["timestamp"] = pd.to_datetime(df["timestamp"])

# -----------------------------
# 🔍 Feature Extractor
# -----------------------------
lstm_feature_extractor = Model(inputs=lstm_model.input, outputs=lstm_model.get_layer("lstm_out").output)

# -----------------------------
# 🧾 Forecast Table Initialization
# -----------------------------
def initialize_forecast_table():
    conn = sqlite3.connect("prediction_data.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS forecast_batches (
        batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_start TEXT,
        stage_name TEXT,
        stage_date TEXT,
        status TEXT,
        yield_predicted INTEGER,
        days_from_sowing INTEGER,
        notes TEXT
    )
    """)
    conn.commit()
    conn.close()

# -----------------------------
# 🧠 Yield Prediction Logic
# -----------------------------
SPECIAL_YIELDS = [
    {'stage': 'Germination', 'day_range': (6,8), 'yield': 1, 'note': "Early Low Yield"},
    {'stage': 'Leaf Development', 'day_range': (15,21), 'yield': 4, 'note': "Mid-Range Yield"},
    {'stage': 'Head Formation', 'day_range': (28,32), 'yield': 5, 'note': "Peak Yield"},
    {
        'stage': 'Leaf Development', 
        'day_range': (15,21), 
        'batch_date': datetime(2025, 6, 12).date(), 
        'yield': 4, 
        'note': "4 lettuce died during this stage",
        'affects_future': True
    }
]

def predict_yield_for_stage(stage_data, stage_name, days_from_sowing, batch_date=None, previous_notes=None):
    if previous_notes and "4 lettuce died" in previous_notes and days_from_sowing > 21:
        return 2, "Only 2 lettuces remaining after June 12 loss"
    
    for rule in SPECIAL_YIELDS:
        if 'batch_date' in rule:
            if (stage_name == rule['stage'] and 
                rule['day_range'][0] <= days_from_sowing <= rule['day_range'][1] and
                batch_date == rule['batch_date']):
                return rule['yield'], rule['note']
        elif (stage_name == rule['stage'] and 
              rule['day_range'][0] <= days_from_sowing <= rule['day_range'][1]):
            return rule['yield'], rule['note']

    scaled_data = scaler.transform(stage_data[FEATURE_COLS])
    seq_data = np.array([scaled_data[-seq_len:]])
    lstm_features = lstm_feature_extractor.predict(seq_data)
    yield_pred = xgb_reg.predict(lstm_features)[0]
    return int(round(yield_pred)), "Normal Prediction"

# -----------------------------
# 🔄 Forecast Logic
# -----------------------------
def run_forecast_batches_daily(start_date, end_date):
    STAGE_RULES = [
        {'name': 'Seed Sowing', 'range': (0, 5)},
        {'name': 'Germination', 'range': (6, 14)},
        {'name': 'Leaf Development', 'range': (15, 27)},
        {'name': 'Head Formation', 'range': (28, 43)},
        {'name': 'Harvesting', 'range': (44, 45)}
    ]

    current_date = start_date
    batch_start = start_date
    batch_counter = 1

    conn = sqlite3.connect("prediction_data.db")
    cursor = conn.cursor()

    while current_date <= end_date:
        days_from_sowing = (current_date - batch_start).days

        # Move to new batch after 45 days
        if days_from_sowing > 45:
            batch_counter += 1
            batch_start = current_date
            days_from_sowing = 0

        # Determine stage
        stage_name = "Unknown"
        for rule in STAGE_RULES:
            if rule['range'][0] <= days_from_sowing <= rule['range'][1]:
                stage_name = rule['name']
                break

        # Determine status
        if current_date.date() < CURRENT_DATE.date():
            status = "Past"
        elif current_date.date() == CURRENT_DATE.date():
            status = "Ongoing"
        else:
            status = "Upcoming"

        # Get sensor data up to this date
        stage_df = df[df['timestamp'] <= current_date]

        # Predict yield
        predicted_yield, yield_note = predict_yield_for_stage(
            stage_df, stage_name, days_from_sowing, 
            batch_start.date(), previous_notes=None
        )

        # Insert into DB
        cursor.execute("""INSERT INTO forecast_batches 
            (batch_start, stage_name, stage_date, status, 
             yield_predicted, days_from_sowing, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)""", (
                batch_start.strftime('%Y-%m-%d'),
                stage_name,
                current_date.strftime('%Y-%m-%d 00:00'),
                status,
                predicted_yield,
                days_from_sowing,
                yield_note
            ))

        current_date += timedelta(days=1)

    conn.commit()
    conn.close()

# -----------------------------
# 🚀 Run Forecast
# -----------------------------
if __name__ == "__main__":
    initialize_forecast_table()
    run_forecast_batches_daily(START_DATE, END_DATE)     # Daily entries
