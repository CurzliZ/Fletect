#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <WiFi.h>
#include <HTTPClient.h>

// ============================================================================
// FLOOD SENTRY — ESP32 Smart Edge Firmware
// IoT-Based Flood Risk Detection Using Deep Neural Networks
// ============================================================================

// Pin Configurations (Preserved exactly matching diagram.json)
const int TRIG_PIN    = 5;   // Ultrasonic TRIG
const int ECHO_PIN    = 18;  // Ultrasonic ECHO
const int RAIN_PIN    = 34;  // Rain Sensor / Potentiometer (ADC1_CH6)
const int FLOW_PIN    = 35;  // Flow Sensor / Potentiometer (ADC1_CH7)
const int LED_GREEN   = 25;  // Green LED: Low Risk / Normal
const int LED_YELLOW  = 26;  // Yellow LED: Medium Risk / Watch
const int LED_RED     = 27;  // Red LED: High Risk / Danger
const int BUZZER_PIN  = 14;  // Acoustic Alert Buzzer
const int ACK_BUTTON  = 13;  // Silence / Acknowledge Button

// Physical Reference Calibration
// SENSOR_TO_DRAIN_BOTTOM_CM: Total height from sensor face to the dry channel bed.
// Water depth = SENSOR_TO_DRAIN_BOTTOM_CM - measured distance.
const float SENSOR_TO_DRAIN_BOTTOM_CM = 100.0;
const char* DEVICE_ID = "ESP32_FLOOD_01";

// Networking configuration for Wokwi Simulation
// Wokwi provides an open virtual WiFi access point named "Wokwi-GUEST".
const char* WIFI_SSID = "Wokwi-GUEST";
const char* WIFI_PASS = "";

// Backend API URL:
// In Wokwi cloud simulation, localhost refers to the container, not your PC.
// If using Wokwi IoT Gateway or a local proxy / ngrok, update this IP.
// The firmware also streams full JSON over Serial for the local Python bridge.
const char* BACKEND_URL = "http://10.0.2.2:8000/api/telemetry"; // Default Wokwi gateway IP

LiquidCrystal_I2C lcd(0x27, 16, 2);

// Telemetry state and non-blocking timers
unsigned long lastTelemetryTime = 0;
const unsigned long TELEMETRY_INTERVAL_MS = 3000;

unsigned long sequenceNumber = 0;
bool alarmAcknowledged = false;
int previousButton = HIGH;
unsigned long lastButtonChange = 0;

float readDistanceCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(3);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // 30,000 microseconds timeout (~5 meters max)
  unsigned long duration = pulseIn(ECHO_PIN, HIGH, 30000UL);
  if (duration == 0) return -1.0;
  return duration * 0.0343 / 2.0;
}

void setLeds(bool green, bool yellow, bool red) {
  digitalWrite(LED_GREEN, green ? HIGH : LOW);
  digitalWrite(LED_YELLOW, yellow ? HIGH : LOW);
  digitalWrite(LED_RED, red ? HIGH : LOW);
}

void sendHttpTelemetry(const String& jsonPayload) {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }

  HTTPClient http;
  http.begin(BACKEND_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(1500);

  int httpCode = http.POST(jsonPayload);
  if (httpCode > 0) {
    Serial.printf("[HTTP] Ingest response code: %d\n", httpCode);
  } else {
    Serial.printf("[HTTP] Post failed: %s (Check network or use serial bridge)\n", http.errorToString(httpCode).c_str());
  }
  http.end();
}

void setup() {
  Serial.begin(115200);
  delay(100);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(ACK_BUTTON, INPUT_PULLUP);

  // Initialize I2C LCD
  Wire.begin(21, 22);
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("FLOOD SENTRY");
  lcd.setCursor(0, 1);
  lcd.print("Connecting WiFi");

  // Initial LED self-test
  setLeds(true, true, true);
  delay(600);
  setLeds(false, false, false);

  // Connect to Wokwi Virtual WiFi
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  unsigned long startAttempt = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startAttempt < 6000) {
    delay(250);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected! IP: " + WiFi.localIP().toString());
    lcd.setCursor(0, 1);
    lcd.print("WiFi Connected  ");
  } else {
    Serial.println("\nWiFi timeout. Running in Serial Telemetry mode.");
    lcd.setCursor(0, 1);
    lcd.print("WiFi Offline    ");
  }
  delay(1000);
  lcd.clear();
  Serial.println("FLOOD SENTRY edge telemetry initialized.");
}

void loop() {
  // Pushbutton debounce & ACK logic
  int buttonNow = digitalRead(ACK_BUTTON);
  if (buttonNow != previousButton) {
    lastButtonChange = millis();
    previousButton = buttonNow;
  }
  if (buttonNow == LOW && millis() - lastButtonChange > 35) {
    alarmAcknowledged = true;
  }

  // Periodic Telemetry and Sensing Loop
  if (millis() - lastTelemetryTime >= TELEMETRY_INTERVAL_MS) {
    lastTelemetryTime = millis();
    sequenceNumber++;

    float distance = readDistanceCm();
    int rainRaw = analogRead(RAIN_PIN);
    int flowRaw = analogRead(FLOW_PIN);

    // Map 12-bit ADC (0-4095) to percentage (0-100%)
    float rainPct = (rainRaw / 4095.0) * 100.0;
    float flowPct = (flowRaw / 4095.0) * 100.0;
    // Map rainfall % to realistic mm/h rate (0-100 mm/h)
    float rainfall_mm_h = rainPct;

    bool validDistance = (distance >= 2.0 && distance <= SENSOR_TO_DRAIN_BOTTOM_CM);
    float waterLevelCm = 0.0;
    float waterLevelPct = 0.0;

    if (validDistance) {
      waterLevelCm = SENSOR_TO_DRAIN_BOTTOM_CM - distance;
      waterLevelPct = (waterLevelCm / SENSOR_TO_DRAIN_BOTTOM_CM) * 100.0;
      waterLevelPct = constrain(waterLevelPct, 0.0, 100.0);
    }

    // Local edge preliminary safety heuristics
    bool danger = validDistance && (waterLevelPct >= 75.0 || (waterLevelPct >= 55.0 && rainPct >= 70.0 && flowPct <= 30.0));
    bool watch = !danger && validDistance && (waterLevelPct >= 40.0 || (rainPct >= 70.0 && flowPct <= 30.0));
    bool normal = !danger && !watch && validDistance;

    // Reset alarm silence if risk state escalates
    static int previousState = -1;
    int currentState = !validDistance ? 3 : danger ? 2 : watch ? 1 : 0;
    if (currentState != previousState) {
      alarmAcknowledged = false;
      previousState = currentState;
    }

    // Update LED indicators
    if (!validDistance) {
      setLeds(false, true, false); // Sensor warning
    } else {
      setLeds(normal, watch, danger);
    }

    // Buzzer alert tone
    if (danger && !alarmAcknowledged) {
      tone(BUZZER_PIN, 1800);
    } else if (watch && !alarmAcknowledged) {
      tone(BUZZER_PIN, 1000, 120);
    } else {
      noTone(BUZZER_PIN);
    }

    // LCD Display Update
    lcd.setCursor(0, 0);
    if (!validDistance) {
      lcd.print("Sensor: NO ECHO ");
      lcd.setCursor(0, 1);
      lcd.print("Check HC-SR04   ");
    } else {
      char line0[17];
      snprintf(line0, sizeof(line0), "Level:%3d%% R:%2d%%", (int)(waterLevelPct + 0.5), (int)(rainPct + 0.5));
      lcd.print(line0);
      lcd.setCursor(0, 1);
      if (danger)      lcd.print("STATUS: DANGER! ");
      else if (watch)  lcd.print("STATUS: WATCH   ");
      else             lcd.print("STATUS: NORMAL  ");
    }

    // Format Telemetry JSON Payload
    String jsonPayload = "{";
    jsonPayload += "\"device_id\":\"" + String(DEVICE_ID) + "\",";
    jsonPayload += "\"seq\":" + String(sequenceNumber) + ",";
    jsonPayload += "\"uptime_s\":" + String(millis() / 1000) + ",";
    jsonPayload += "\"distance_cm\":" + String(distance, 1) + ",";
    jsonPayload += "\"water_level_cm\":" + String(waterLevelCm, 1) + ",";
    jsonPayload += "\"water_level_pct\":" + String(waterLevelPct, 1) + ",";
    jsonPayload += "\"rainfall_raw\":" + String(rainRaw) + ",";
    jsonPayload += "\"rainfall_pct\":" + String(rainPct, 1) + ",";
    jsonPayload += "\"rainfall_mm_h\":" + String(rainfall_mm_h, 1) + ",";
    jsonPayload += "\"flow_raw\":" + String(flowRaw) + ",";
    jsonPayload += "\"flow_pct\":" + String(flowPct, 1) + ",";
    jsonPayload += "\"channel_capacity_cm\":" + String(SENSOR_TO_DRAIN_BOTTOM_CM, 1) + ",";
    jsonPayload += "\"valid_sensor\":" + String(validDistance ? "true" : "false") + ",";
    jsonPayload += "\"data_source\":\"simulation\"";
    jsonPayload += "}";

    // 1. Emit structured telemetry line to Serial
    Serial.print("TELEMETRY:");
    Serial.println(jsonPayload);

    // 2. Dispatch via HTTP POST if connected
    sendHttpTelemetry(jsonPayload);
  }

  delay(50);
}
