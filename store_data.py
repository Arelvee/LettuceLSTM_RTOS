import sqlite3
import serial
from datetime import datetime

# Setup serial port
ser = serial.Serial('COM4', 115200, timeout=1)

# SQLite DB connection
conn = sqlite3.connect('sensor_data.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS sensor_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        humidity REAL,
        temp_env REAL,
        temp_water REAL,
        ec REAL,
        tds REAL,
        light REAL,
        ppfd REAL,
        reflect_445 INTEGER,
        reflect_480 INTEGER
    )
''')
conn.commit()

try:
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line and "," in line:
            parts = line.split(",")
            if len(parts) == 10:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                data = [timestamp] + [float(p) for p in parts[1:8]] + [int(parts[8]), int(parts[9])]
                cursor.execute('''
                    INSERT INTO sensor_logs (
                        timestamp, humidity, temp_env, temp_water,
                        ec, tds, light, ppfd, reflect_445, reflect_480
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', data)
                conn.commit()
                print(f"📥 Inserted into DB: {data}")
except KeyboardInterrupt:
    print("🛑 Stopped logging.")
    conn.close()
