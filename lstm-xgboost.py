# [0] Imports & Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, mean_absolute_error, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import joblib
import shap
from datetime import datetime
from collections import Counter
import optuna
from optuna.integration import XGBoostPruningCallback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize



# [1] Load & Preprocess Data
def load_and_preprocess_data(filepath, seq_len=10):
    df = pd.read_csv(filepath)

    # Extract targets
    original_labels = df['growth_stage'].values
    y_yield = df['yield_count'].values

    # Drop unused columns and scale features
    features = df.drop(columns=['yield_count', 'growth_stage', 'timestamp', 'batch_id'], errors='ignore')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    # Sequence creation
    def create_sequences(data, length):
        return np.array([data[i:i + length] for i in range(len(data) - length)])

    def create_targets(target, length):
        return np.array([target[i + length] for i in range(len(target) - length)])

    X_seq = create_sequences(scaled, seq_len)
    y_yield_seq = create_targets(y_yield, seq_len)
    y_stage_seq = create_targets(original_labels, seq_len)

    # Encode labels using fixed order
    fixed_stage_order = ['Seed Sowing', 'Germination', 'Leaf Development', 'Head Formation', 'Harvesting']
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(fixed_stage_order)
    y_stage_encoded = label_encoder.transform(y_stage_seq)

    return X_seq, y_yield_seq, y_stage_encoded, label_encoder, scaler

# [2] Build LSTM Models
def build_lstm_feature_model(seq_len, n_features):
  # Fixed batch size for TFLite Micro: batch=1
    inp = Input(batch_shape=(1, seq_len, n_features), name="input")
    x = LSTM(128, return_sequences=False, name="lstm_out")(inp)
    out = Dense(1, name="yield_output")(x)
    model = Model(inputs=inp, outputs=out)
    return model

def build_forecast_model(seq_len, n_features):
    inp = Input(batch_shape=(1, seq_len, n_features), name="input")
    x = LSTM(128, return_sequences=False, name="lstm_forecast")(inp)
    out = Dense(n_features, name="sensor_output")(x)
    model = Model(inputs=inp, outputs=out)
    return model

# [3] Confusion Matrix Plot
def plot_confusion_matrix(y_true, y_pred, labels, fname="conf_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.matshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center", color="black")
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Growth Stage Confusion Matrix")
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()

# [4] LSTM Feature Extraction + Train/Test Split
SEQ_LEN = 10
X, y_yield, y_stage, le, scaler = load_and_preprocess_data("lettuce_wavelet_cleaned.csv", SEQ_LEN)
class_names = le.classes_

X_train, X_test, y_yield_train, y_yield_test, y_stage_train, y_stage_test = train_test_split(
    X, y_yield, y_stage, test_size=0.1, stratify=y_stage, random_state=42
)
print("Training Set Growth Stage Distribution:")
for label_idx, count in sorted(Counter(y_stage_train).items()):
    print(f"{class_names[label_idx]} ({label_idx}): {count}")
class_names = list(le.classes_) 
lstm_model = build_lstm_feature_model(SEQ_LEN, X.shape[2])
lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
lstm_model.fit(X_train, y_yield_train, validation_data=(X_test, y_yield_test),
               epochs=50, batch_size=32, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
lstm_model.summary()

feature_extractor = Model(inputs=lstm_model.input, outputs=lstm_model.get_layer("lstm_out").output)
X_train_lstm = feature_extractor.predict(X_train)
X_test_lstm = feature_extractor.predict(X_test)

# [5] XGBoost Regression & Classification
xgb_reg = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
xgb_reg.fit(X_train_lstm, y_yield_train)
y_pred_yield = xgb_reg.predict(X_test_lstm)

# Compute class weights: inverse of class frequency
class_counts = Counter(y_stage_train)
total = sum(class_counts.values())
class_weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}

# Apply weights to training data
# y_stage_train is your encoded target (integer labels from LabelEncoder)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_stage_train)

xgb_clf = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(class_names),
    eval_metric="mlogloss"
)
xgb_clf.fit(X_train_lstm, y_stage_train, sample_weight=sample_weights)

print("\n📊 XGBoost Regressor Parameters:")
print(xgb_reg.get_params())
print("Feature Importances (Regressor):", xgb_reg.feature_importances_)
print("\n📊 XGBoost Classifier Parameters:")
print(xgb_clf.get_params())
print("Feature Importances (Classifier):", xgb_clf.feature_importances_)

# [6] Evaluation
y_pred_stage = xgb_clf.predict(X_test_lstm)

print(f"Yield MAE: {mean_absolute_error(y_yield_test, y_pred_yield):.4f}")
print("\nGrowth Stage Report:\n", classification_report(
    y_stage_test,
    y_pred_stage,
    target_names=class_names  # safe because y_stage_test and y_pred_stage are integers
))
# Binarize the ground truth and predictions for mAP calculation
y_stage_test_bin = label_binarize(y_stage_test, classes=list(range(len(class_names))))
y_pred_stage_bin = label_binarize(y_pred_stage, classes=list(range(len(class_names))))
map_score = average_precision_score(y_stage_test_bin, y_pred_stage_bin, average='macro')
print(f"\n🎯 Mean Average Precision (mAP): {map_score:.4f}")

plot_confusion_matrix(y_stage_test, y_pred_stage, labels=list(range(len(class_names))), fname="conf_matrix.png")


print("\n--- Stratified K-Fold ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = []

# ✅ Initialize cumulative confusion matrix
cumulative_cm = np.zeros((len(class_names), len(class_names)), dtype=int)

for i, (train_idx, val_idx) in enumerate(skf.split(X, y_stage), 1):
    X_train_f, X_val_f = X[train_idx], X[val_idx]
    y_y_train_f, y_s_train_f = y_yield[train_idx], y_stage[train_idx]
    y_s_val_f = y_stage[val_idx]

    # 🔁 Train LSTM
    fold_model = build_lstm_feature_model(SEQ_LEN, X.shape[2])
    fold_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    fold_model.fit(X_train_f, y_y_train_f, epochs=50, batch_size=32, verbose=0,
                   callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

    # 🔍 Extract features
    extractor = Model(fold_model.input, fold_model.get_layer("lstm_out").output)
    X_lstm_f_train = extractor.predict(X_train_f)
    X_lstm_f_val = extractor.predict(X_val_f)

    # 🌲 Train XGBoost classifier
    clf = xgb.XGBClassifier(objective="multi:softmax", num_class=len(class_names), eval_metric="mlogloss")
    sample_weights_fold = compute_sample_weight(class_weight='balanced', y=y_s_train_f)
    clf.fit(X_lstm_f_train, y_s_train_f, sample_weight=sample_weights_fold)
    y_pred_fold = clf.predict(X_lstm_f_val)

    # 🧪 Evaluate
    y_val_bin = label_binarize(y_s_val_f, classes=list(range(len(class_names))))
    y_pred_bin = label_binarize(y_pred_fold, classes=list(range(len(class_names))))

    metrics.append({
        "acc": accuracy_score(y_s_val_f, y_pred_fold),
        "prec": precision_score(y_s_val_f, y_pred_fold, average='weighted', zero_division=0),
        "rec": recall_score(y_s_val_f, y_pred_fold, average='weighted', zero_division=0),
        "map": average_precision_score(y_val_bin, y_pred_bin, average='macro')
    })

    # 📊 Accumulate confusion matrix
    cm = confusion_matrix(y_s_val_f, y_pred_fold, labels=list(range(len(class_names))))
    cumulative_cm += cm

# 🖨️ Print per-fold metrics
for i, m in enumerate(metrics, 1):
    print(f"Fold {i}: Acc={m['acc']:.4f}, Prec={m['prec']:.4f}, Rec={m['rec']:.4f}, mAP={m['map']:.4f}")

# 📈 Average confusion matrix
avg_cm = cumulative_cm / skf.get_n_splits()

# 📊 Plot average confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(avg_cm, cmap="Blues")
fig.colorbar(cax)

for (i, j), val in np.ndenumerate(avg_cm):
    ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="black")

plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=45)
plt.yticks(ticks=np.arange(len(class_names)), labels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Average Confusion Matrix (5-Fold)")
plt.tight_layout()
plt.savefig("avg_conf_matrix_kfold.png")
plt.show()

# 📌 Optional: print average metrics across all folds
avg_metrics = {
    key: np.mean([m[key] for m in metrics])
    for key in ["acc", "prec", "rec", "map"]
}
print(f"\nAvg: Acc={avg_metrics['acc']:.4f}, Prec={avg_metrics['prec']:.4f}, Rec={avg_metrics['rec']:.4f}, mAP={avg_metrics['map']:.4f}")

import os
import pickle

# Create directory for model files
os.makedirs("saved_models", exist_ok=True)

# Save LSTM model
lstm_model.save("saved_models/lstm_feature_extractor.keras")

# Save XGBoost models using native save_model (more reliable than pickle/joblib)
xgb_reg.save_model("saved_models/xgb_reg.json")
xgb_clf.save_model("saved_models/xgb_clf.json")

# Save label encoder and scaler with pickle
with open("saved_models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

with open("saved_models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ All models saved safely in 'saved_models/' directory.")

# [9] Forecasting Sensor Data
def prepare_forecast_data(df, seq_len=10):
    df_sensors = df.drop(columns=['yield_count', 'growth_stage', 'timestamp', 'batch_id'], errors='ignore')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_sensors)

    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len])
    return np.array(X), np.array(y), scaler

df_raw = pd.read_csv("lettuce_wavelet_cleaned.csv")
X_fcast, y_fcast, fcast_scaler = prepare_forecast_data(df_raw, SEQ_LEN)

X_f_train, X_f_test, y_f_train, y_f_test = train_test_split(X_fcast, y_fcast, test_size=0.1, random_state=42)
forecast_model = build_forecast_model(SEQ_LEN, X_fcast.shape[2])
forecast_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
forecast_model.fit(X_f_train, y_f_train, validation_data=(X_f_test, y_f_test), epochs=30,
                   batch_size=32, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
mae_fcast = forecast_model.evaluate(X_f_test, y_f_test)[1]
forecast_model.summary()
forecast_model.save("sensor_forecast_model.keras")
joblib.dump(fcast_scaler, "sensor_forecast_scaler.save")
print(f"\n📈 Sensor Forecast MAE: {mae_fcast:.4f}")

# [10] Optuna Hyperparameter Tuning
def objective(trial):
    params = {
        "objective": "multi:softmax",
        "num_class": len(class_names),
        "eval_metric": "mlogloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "verbosity": 0,
        "random_state": 42
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
    X_train_lstm,
    y_stage_train,
    sample_weight=sample_weights,
    eval_set=[(X_test_lstm, y_stage_test)],
    verbose=False
)

    preds = model.predict(X_test_lstm)
    return accuracy_score(y_stage_test, preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print(f"\n🎯 Best Accuracy: {study.best_value:.4f}")
print("Best Parameters:", study.best_params)

# [11] Retrain with Best Params + SHAP
best_model = xgb.XGBClassifier(**study.best_params)
best_model.fit(X_train_lstm, y_stage_train)
joblib.dump(best_model, "xgb_stage_optuna.pkl")

explainer = shap.Explainer(best_model)
shap_values = explainer(X_test_lstm[:100])
shap.summary_plot(shap_values, X_test_lstm[:100], feature_names=[f"LSTM_{i}" for i in range(X_test_lstm.shape[1])])
