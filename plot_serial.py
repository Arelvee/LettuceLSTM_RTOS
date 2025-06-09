import serial
import matplotlib.pyplot as plt
from collections import deque
import sqlite3
from datetime import datetime

# === Settings ===
PORT = "COM4"  # Adjust as needed
BAUD_RATE = 115200
MAX_POINTS = 100

# === SQLite setup ===
conn = sqlite3.connect("sensor_readings.db")
cursor = conn.cursor()
cursor.execute(''' 
    CREATE TABLE IF NOT EXISTS sensor_readings (
        timestamp TEXT,
        humidity REAL,
        temp_envi REAL,
        temp_water REAL,
        tds REAL,
        lux REAL,
        ppfd REAL,
        reflect_445 REAL,
        reflect_480 REAL,
        ph REAL
    )
''')
conn.commit()

# === Data containers ===
humidity_vals = deque(maxlen=MAX_POINTS)
temp_envi_vals = deque(maxlen=MAX_POINTS)
temp_water_vals = deque(maxlen=MAX_POINTS)
tds_vals = deque(maxlen=MAX_POINTS)
lux_vals = deque(maxlen=MAX_POINTS)
ppfd_vals = deque(maxlen=MAX_POINTS)
reflect_445_vals = deque(maxlen=MAX_POINTS)
reflect_480_vals = deque(maxlen=MAX_POINTS)
ph_vals = deque(maxlen=MAX_POINTS)

# === Plot setup ===
plt.ion()
fig, ax = plt.subplots()
lines = [
    ax.plot([], [], label="Humidity (%)")[0],
    ax.plot([], [], label="Temp (Envi °C)")[0],
    ax.plot([], [], label="Temp (Water °C)")[0],
    ax.plot([], [], label="TDS (ppm)")[0],
    ax.plot([], [], label="Light (lux)")[0],
    ax.plot([], [], label="PPFD (µmol/m²/s)")[0],
    ax.plot([], [], label="Reflectance 445 nm")[0],
    ax.plot([], [], label="Reflectance 480 nm")[0],
    ax.plot([], [], label="pH")[0]
]
ax.set_ylim(0, 500)  # Adjust as needed
ax.legend(loc='upper left')

# === Open serial port ===
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)

try:
    while True:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            values = line.split(",")

            if len(values) == 10:
                humidity = float(values[1])
                temp_envi = float(values[2])
                temp_water = float(values[3])
                tds = float(values[4])
                lux = float(values[5])
                ppfd = float(values[6])
                reflect_445 = float(values[7])
                reflect_480 = float(values[8])
                ph = float(values[9])
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Insert into DB
                cursor.execute('''
                    INSERT INTO sensor_readings 
                    (timestamp, humidity, temp_envi, temp_water, tds, lux, ppfd, reflect_445, reflect_480, ph)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (timestamp, humidity, temp_envi, temp_water, tds, lux, ppfd, reflect_445, reflect_480, ph))
                conn.commit()

                # Append to buffers
                humidity_vals.append(humidity)
                temp_envi_vals.append(temp_envi)
                temp_water_vals.append(temp_water)
                tds_vals.append(tds)
                lux_vals.append(lux)
                ppfd_vals.append(ppfd)
                reflect_445_vals.append(reflect_445)
                reflect_480_vals.append(reflect_480)
                ph_vals.append(ph)

                # Plot
                all_vals = [
                    humidity_vals, temp_envi_vals, temp_water_vals, tds_vals,
                    lux_vals, ppfd_vals, reflect_445_vals, reflect_480_vals, ph_vals
                ]
                x = range(len(humidity_vals))
                for line_plot, data in zip(lines, all_vals):
                    line_plot.set_xdata(x)
                    line_plot.set_ydata(data)

                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.01)

            else:
                print("⚠️ Invalid data format:", line)

        except ValueError:
            print("⚠️ Value error in line:", line)
        except Exception as e:
            print("⚠️ Unexpected error:", e)

except KeyboardInterrupt:
    print("🛑 Stopped by user.")
finally:
    ser.close()
    conn.close()
    print("✅ Serial and DB connections closed.")
