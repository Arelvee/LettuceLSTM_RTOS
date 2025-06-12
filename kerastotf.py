import tensorflow as tf

SEQ_LEN = 10
N_FEATURES = 18  # adjust to your dataset

# Create and build model
forecast_model = build_forecast_model(SEQ_LEN, N_FEATURES)
forecast_model.compile(optimizer="adam", loss="mse")
forecast_model.build(input_shape=(1, SEQ_LEN, N_FEATURES))  # Fix the input shape
forecast_model.summary()

# Save
forecast_model.save("lstm_feature_extractor.keras")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(forecast_model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter._experimental_lower_tensor_list_ops = True
tflite_model = converter.convert()

# Save the .tflite model
with open("sensor_forecast_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ sensor_forecast_model.tflite ready for ESP32")
