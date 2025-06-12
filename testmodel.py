import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model, Model
from xgboost import XGBRegressor, XGBClassifier
import pickle



def initialize_forecast_table():
    conn = sqlite3.connect("prediction_data.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS forecast_batches (
        batch_start TEXT,
        stage_name TEXT,
        stage_date TEXT,
        status TEXT,
        yield_predicted INTEGER,
        yield_date TEXT,
        yield_status TEXT
    )
    """)
    conn.commit()
    conn.close()


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
END_DATE = datetime(2025, 8, 15)
BATCH_DURATION = timedelta(days=46)

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
# 🔄 Forecast Logic
# -----------------------------
def run_forecast_batches(start_date, end_date):
    STAGES = ['Seed Sowing', 'Germination', 'Leaf Development', 'Head Formation', 'Harvesting']
    current_batch = start_date

    while current_batch <= end_date:
        # Fixed stage schedule for 46-day cycle
        batch_dates = {
            'Seed Sowing': current_batch,
            'Germination': current_batch + timedelta(days=7),
            'Leaf Development': current_batch + timedelta(days=18),
            'Head Formation': current_batch + timedelta(days=30),
            'Harvesting': current_batch + timedelta(days=44)
        }

        predicted_yield = (current_batch + timedelta(days=45), 6)

        # Use May 31, 2025 as the fixed "reference" today
        fixed_today = datetime(2025, 5, 31)

        def status(date):
            if date.date() < fixed_today.date():
                return "(✅ Past)"
            elif date.date() == fixed_today.date():
                return "(⏳ Ongoing)"
            else:
                return "(📅 Upcoming)"

        # Output for the batch
        print("")
        for stage in STAGES:
            date = batch_dates[stage]
            icon = "🟢" if date < fixed_today else ("🟡" if date.date() == fixed_today.date() else "🔜")
            print(f"{icon} {stage}: {date.strftime('%Y-%m-%d 00:00')} {status(date)}")

        y_date, y_val = predicted_yield
        y_status = "✅ Past" if y_date < fixed_today else ("⏳ Ongoing" if y_date.date() == fixed_today.date() else "🔜 Upcoming")
        print(f"📈 Yield ≥ 6: {y_date.strftime('%Y-%m-%d 00:00')} ({y_status}, Predicted: {y_val})")

        current_batch += timedelta(days=46)



# -----------------------------
# 🚀 Run Forecast
# -----------------------------
if __name__ == "__main__":
    run_forecast_batches(START_DATE, END_DATE)