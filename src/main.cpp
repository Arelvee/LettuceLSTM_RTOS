#include <WiFi.h>
#include <NTPClient.h>
#include <WiFiUdp.h>
#include <DHT.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Wire.h>
#include <BH1750.h>
#include <Adafruit_AS7341.h>

// WiFi credentials
const char *ssid = "Redmi Note 13 Pro";
const char *password = "#FJLMiloves0225";

// DHT Sensor
#define DHTPIN 19
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

// Dallas Temperature Sensor
#define ONE_WIRE_BUS 18
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

// Light Sensor
BH1750 lightMeter;

// TDS Sensor
#define TdsSensorPin 32
float tdsValue = 0, voltage = 0, ecValue = 0;

// pH Sensor
#define pHSensorPin 35
float pHValue = 0;

// AS7341
Adafruit_AS7341 as7341;

// NTP
WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP, "asia.pool.ntp.org", 3600 * 8, 60000); // Manila Time

// Global Shared Data
float humidity = 0, tempDHT = 0, tempDallas = 0;
float lightIntensity = 0, ppfd = 0;
uint16_t reflect_445 = 0, reflect_480 = 0;

// Data structure for ML
typedef struct {
  char timeString[9];
  float humidity;
  float tempDHT;
  float tempDallas;
  float ecValue;
  float tdsValue;
  float lightIntensity;
  float ppfd;
  float pHValue;
  uint16_t reflect_445;
  uint16_t reflect_480;
} SensorData_t;

// FreeRTOS Queue
QueueHandle_t sensorDataQueue;

// Function Prototypes
void connectToWiFi();
void taskReadDHT(void *pvParameters);
void taskReadDallas(void *pvParameters);
void taskReadTDS(void *pvParameters);
void taskReadPH(void *pvParameters);
void taskReadAS7341(void *pvParameters);
void taskReadLight(void *pvParameters);
void taskMonitor(void *pvParameters);
void taskMLProcessor(void *pvParameters);

// WiFi Connection
void connectToWiFi() {
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 20) {
    delay(500);
    Serial.print(".");
    retries++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ WiFi Connected.");
    Serial.print("📶 IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n❌ Failed to connect to WiFi.");
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  dht.begin();
  sensors.begin();
  Wire.begin();
  lightMeter.begin();

  if (!as7341.begin()) {
    Serial.println("AS7341 not detected. Check wiring.");
    while (1);
  }

  connectToWiFi();
  timeClient.begin();

  sensorDataQueue = xQueueCreate(20, sizeof(SensorData_t)); // Queue size of 20

  // Sensor tasks - Core 0 (sensor reading)
  xTaskCreatePinnedToCore(taskReadDHT, "Read DHT", 2048, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(taskReadDallas, "Read Dallas", 2048, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(taskReadTDS, "Read TDS", 2048, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(taskReadPH, "Read pH", 2048, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(taskReadAS7341, "Read AS7341", 3072, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(taskReadLight, "Read Light", 2048, NULL, 1, NULL, 0);

  // Processing and monitoring tasks - Core 1
  xTaskCreatePinnedToCore(taskMonitor, "Monitor", 3072, NULL, 1, NULL, 1);
  xTaskCreatePinnedToCore(taskMLProcessor, "ML Processor", 8192, NULL, 2, NULL, 1);
}

void taskReadDHT(void *pvParameters) {
  for (;;) {
    float h = dht.readHumidity();
    float t = dht.readTemperature();
    if (!isnan(h) && !isnan(t)) {
      humidity = h;
      tempDHT = t;
    } else {
      Serial.println("[DHT Task] ❌ Failed to read from DHT sensor!");
    }
    vTaskDelay(2000 / portTICK_PERIOD_MS);
  }
}

void taskReadDallas(void *pvParameters) {
  for (;;) {
    sensors.requestTemperatures();
    float t = sensors.getTempCByIndex(0);
    if (t != DEVICE_DISCONNECTED_C) {
      tempDallas = t;
    } else {
      Serial.println("[Dallas Task] ❌ Sensor disconnected!");
    }
    vTaskDelay(2000 / portTICK_PERIOD_MS);
  }
}

void taskReadTDS(void *pvParameters) {
  const float calibrationEC = 1413.0;
  const float calibrationVoltage = 1.07;
  const float ecCorrectionFactor = 1.54;
  const float tdsConversionFactor = 0.49;

  for (;;) {
    int analogValue = analogRead(TdsSensorPin);
    voltage = analogValue * (3.3 / 4095.0);
    float ecRaw = (voltage / calibrationVoltage) * calibrationEC;
    float tempCompensation = 1.0 + 0.02 * (tempDallas - 25.0);
    ecValue = (ecRaw / tempCompensation) * ecCorrectionFactor;
    tdsValue = ecValue * tdsConversionFactor;

    Serial.printf("[TDS Task] Voltage: %.2f V | EC: %.2f µS/cm | TDS: %.2f ppm\n", voltage, ecValue, tdsValue);
    vTaskDelay(2000 / portTICK_PERIOD_MS);
  }
}

void taskReadPH(void *pvParameters) {
  const float calibrationMidVoltage = 1.65;  // Voltage at pH 7.0
  const float slope = -5.7;                  // Adjust this based on your calibration

  for (;;) {
    int adcValue = analogRead(pHSensorPin);
    float voltage = adcValue * (3.3 / 4095.0);
    float rawPH = 7 + ((voltage - calibrationMidVoltage) * slope);

    // Clamp pH
    if (rawPH < 6.299) rawPH = 6.299;
    else if (rawPH > 6.7999) rawPH = 6.7999;

    pHValue = rawPH;
    Serial.printf("[pH Task] Voltage: %.2f V | pH: %.4f\n", voltage, pHValue);
    vTaskDelay(2000 / portTICK_PERIOD_MS);
  }
}

void taskReadAS7341(void *pvParameters) {
  for (;;) {
    if (as7341.readAllChannels()) {
      reflect_445 = as7341.getChannel(AS7341_CHANNEL_445nm_F2);
      reflect_480 = as7341.getChannel(AS7341_CHANNEL_480nm_F3);
      Serial.printf("[AS7341 Task] Reflectance @445nm: %u | @480nm: %u\n", reflect_445, reflect_480);
    } else {
      Serial.println("[AS7341 Task] ❌ Failed to read AS7341 data.");
    }
    vTaskDelay(2000 / portTICK_PERIOD_MS);
  }
}

void taskReadLight(void *pvParameters) {
  for (;;) {
    float lux = lightMeter.readLightLevel();
    if (lux >= 0) {
      lightIntensity = lux;
    } else {
      Serial.println("[Light Task] ❌ Failed to read BH1750 sensor!");
    }
    vTaskDelay(2000 / portTICK_PERIOD_MS);
  }
}

void taskMonitor(void *pvParameters) {
  for (;;) {
    SensorData_t data;

    if (WiFi.status() == WL_CONNECTED) timeClient.update();

    sprintf(data.timeString, "%02d:%02d:%02d", timeClient.getHours(), timeClient.getMinutes(), timeClient.getSeconds());

    data.humidity = humidity;
    data.tempDHT = tempDHT;
    data.tempDallas = tempDallas;
    data.ecValue = ecValue;
    data.tdsValue = tdsValue;
    data.lightIntensity = lightIntensity;
    data.ppfd = lightIntensity * 0.0185;
    data.pHValue = pHValue;
    data.reflect_445 = reflect_445;
    data.reflect_480 = reflect_480;

    xQueueSend(sensorDataQueue, &data, portMAX_DELAY);

    Serial.printf("[Monitor] %s, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.4f, %u, %u\n",
      data.timeString, data.humidity, data.tempDHT, data.tempDallas,
      data.ecValue, data.tdsValue, data.lightIntensity, data.ppfd,
      data.pHValue, data.reflect_445, data.reflect_480);

    Serial.println("=======================================");
    vTaskDelay(5000 / portTICK_PERIOD_MS);
  }
}

#define WINDOW_SIZE 10
SensorData_t windowBuffer[WINDOW_SIZE];

void taskMLProcessor(void *pvParameters) {
  int index = 0;
  for (;;) {
    SensorData_t incoming;
    if (xQueueReceive(sensorDataQueue, &incoming, portMAX_DELAY) == pdPASS) {
      for (int i = 0; i < WINDOW_SIZE - 1; i++)
        windowBuffer[i] = windowBuffer[i + 1];

      windowBuffer[WINDOW_SIZE - 1] = incoming;

      if (++index >= WINDOW_SIZE) {
        Serial.println("[ML Task] 🔍 Running LSTM/XGBoost inference...");
        // TODO: Add ML inference logic here
        // predict(windowBuffer);
        index = WINDOW_SIZE - 1;
      }
    }
  }
}

void loop() {
  delay(1000); // Idle loop
}
