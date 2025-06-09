import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, mean_absolute_error, confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
from xgboost import XGBClassifier
import joblib
from tensorflow.keras.utils import model_to_dot
from io import StringIO

# Load and preprocess data
df = pd.read_csv("lettuce_wavelet_cleaned.csv")
original_labels = df['growth_stage'].values
y_yield = df['yield_count'].values

X_features = df.drop(columns=['yield_count', 'growth_stage', 'timestamp', 'batch_id'], errors='ignore')

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(X_features)

# Sequence creation
SEQ_LEN = 10

def create_sequences(data, seq_len):
    return np.array([data[i:i + seq_len] for i in range(len(data) - seq_len)])

def create_target_sequences(target, seq_len):
    return np.array([target[i + seq_len] for i in range(len(target) - seq_len)])

X_seq = create_sequences(scaled_data, SEQ_LEN)
y_yield_seq = create_target_sequences(y_yield, SEQ_LEN)
y_stage_seq = create_target_sequences(original_labels, SEQ_LEN)

# Encode growth stage
label_encoder = LabelEncoder()
y_stage_encoded = label_encoder.fit_transform(y_stage_seq)
class_names = label_encoder.classes_

# Train/test split
X_train_seq, X_test_seq, y_yield_train, y_yield_test, y_stage_train, y_stage_test = train_test_split(
    X_seq, y_yield_seq, y_stage_encoded, test_size=0.1, random_state=42, stratify=y_stage_encoded
)

# Build LSTM feature extractor with regression output (1 output)
input_layer = Input(shape=(SEQ_LEN, X_seq.shape[2]))
x = LSTM(128, return_sequences=False, name='lstm_out')(input_layer)
output = Dense(1, name='yield_output')(x)
lstm_model = Model(inputs=input_layer, outputs=output)

lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

# Train LSTM model for yield prediction directly
lstm_model.fit(X_train_seq, y_yield_train, epochs=30, batch_size=32, callbacks=[early_stop], verbose=1)

# Extract features from the LSTM's last LSTM layer output (before Dense)
feature_extractor = Model(inputs=lstm_model.input, outputs=lstm_model.get_layer('lstm_out').output)

X_train_lstm = feature_extractor.predict(X_train_seq)
X_test_lstm = feature_extractor.predict(X_test_seq)

# Print LSTM Model Summary
print("\n--- LSTM Model Summary ---")
lstm_model.summary()

# --- XGBoost: Yield (regression)
xgb_yield = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_yield.fit(X_train_lstm, y_yield_train)
y_yield_pred = xgb_yield.predict(X_test_lstm)

# --- XGBoost: Growth Stage (classification)
xgb_stage = xgb.XGBClassifier(objective='multi:softmax', num_class=len(class_names), eval_metric='mlogloss', random_state=42)
xgb_stage.fit(X_train_lstm, y_stage_train)
y_stage_pred = xgb_stage.predict(X_test_lstm)

# Decode predictions
y_stage_test_decoded = label_encoder.inverse_transform(y_stage_test)
y_stage_pred_decoded = label_encoder.inverse_transform(y_stage_pred)

# --- Evaluation for Yield
mae = mean_absolute_error(y_yield_test, y_yield_pred)
print(f"\nYield Prediction MAE: {mae:.4f}")

# --- Evaluation for Growth Stage Classification
print("\nGrowth Stage Classification Report:")
print(classification_report(y_stage_test_decoded, y_stage_pred_decoded, target_names=class_names))

# Accuracy, Precision, Recall
accuracy = accuracy_score(y_stage_test, y_stage_pred)
precision = precision_score(y_stage_test, y_stage_pred, average='weighted', zero_division=0)
recall = recall_score(y_stage_test, y_stage_pred, average='weighted', zero_division=0)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Mean Average Precision (mAP) approximation (macro average precision)
mAP = precision_score(y_stage_test, y_stage_pred, average='macro', zero_division=0)
print(f"Mean Average Precision (mAP): {mAP:.4f}")


# Print XGBoost Summary
print("\n--- XGBoost Model Summary ---")
print(xgb_yield.get_params())

# Confusion Matrix plot
conf_mat = confusion_matrix(y_stage_test_decoded, y_stage_pred_decoded, labels=class_names)
plt.figure(figsize=(10, 6))
plt.imshow(conf_mat, cmap='Blues')
plt.colorbar()
plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
plt.yticks(np.arange(len(class_names)), class_names)
plt.title("Growth Stage Confusion Matrix")
plt.tight_layout()
plt.savefig("results_lstm_xgboost.png")
plt.show()

# --- Cross Fold Validation for Growth Stage Classification ---
print("\n--- Stratified K-Fold Cross Validation on Growth Stage Classifier ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
accuracies, precisions, recalls, mAPs = [], [], [], []

for train_index, val_index in skf.split(X_seq, y_stage_encoded):
    X_train_f, X_val_f = X_seq[train_index], X_seq[val_index]
    y_stage_train_f, y_stage_val_f = y_stage_encoded[train_index], y_stage_encoded[val_index]
    y_yield_train_f = y_yield_seq[train_index]
    
    # Train LSTM feature extractor (reuse same architecture)
    lstm_model_fold = Model(inputs=input_layer, outputs=output)
    lstm_model_fold.compile(optimizer='adam', loss='mse', metrics=['mae'])
    lstm_model_fold.fit(X_train_f, y_yield_train_f, epochs=30, batch_size=32, verbose=0)
    
    feature_extractor_fold = Model(inputs=lstm_model_fold.input, outputs=lstm_model_fold.get_layer('lstm_out').output)
    
    X_train_lstm_f = feature_extractor_fold.predict(X_train_f)
    X_val_lstm_f = feature_extractor_fold.predict(X_val_f)
    
    # Train XGB classifier
    xgb_stage_fold = xgb.XGBClassifier(objective='multi:softmax', num_class=len(class_names), eval_metric='mlogloss', use_label_encoder=False, random_state=42)
    xgb_stage_fold.fit(X_train_lstm_f, y_stage_train_f)
    
    y_val_pred = xgb_stage_fold.predict(X_val_lstm_f)
    
    acc_f = accuracy_score(y_stage_val_f, y_val_pred)
    prec_f = precision_score(y_stage_val_f, y_val_pred, average='weighted', zero_division=0)
    rec_f = recall_score(y_stage_val_f, y_val_pred, average='weighted', zero_division=0)
    map_f = precision_score(y_stage_val_f, y_val_pred, average='macro', zero_division=0)

    
    accuracies.append(acc_f)
    precisions.append(prec_f)
    recalls.append(rec_f)
    mAPs.append(map_f)
    
    print(f"Fold {fold} - Accuracy: {acc_f:.4f}, Precision: {prec_f:.4f}, Recall: {rec_f:.4f}, mAP: {map_f:.4f}")
    fold += 1

print(f"\nAverage Accuracy: {np.mean(accuracies):.4f}")
print(f"Average Precision: {np.mean(precisions):.4f}")
print(f"Average Recall: {np.mean(recalls):.4f}")
print(f"Average mAP: {np.mean(mAPs):.4f}")

# Save models and encoders
lstm_model.save("lstm_feature_extractor.keras")
joblib.dump(scaler, "scaler.save")
joblib.dump(label_encoder, "label_encoder.save")
joblib.dump(xgb_yield, "xgb_yield_model.pkl")
joblib.dump(xgb_stage, "xgb_stage_model.pkl")
print("\n✅ Models and artifacts saved successfully.")
