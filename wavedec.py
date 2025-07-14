# Finalized Wavelet Denoising Script with Combined cA/cD Coefficients CSV Output (Raw Coeffs, Not Reconstructed)

import os
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from matplotlib.backends.backend_pdf import PdfPages

# Load dataset
df = pd.read_csv("lettuce_growth_data.csv", parse_dates=["timestamp"])

sensor_columns = [
    "humidity", "temp_envi", "temp_water", "tds", "ec",
    "lux", "ppfd", "reflect_445", "reflect_480", "ph"
]

# Sensor configuration (formalized)
sensor_config_table = pd.DataFrame([
    ["humidity",      "db8",  5, "soft"],
    ["temp_envi",     "db4",  3, "hard"],
    ["temp_water",    "db4",  3, "hard"],
    ["tds",           "db4",  4, "soft"],
    ["ec",            "db4",  4, "soft"],
    ["lux",           "db4",  4, "soft"],
    ["ppfd",          "db4",  4, "soft"],
    ["reflect_445",   "sym8", 6, "garrote"],
    ["reflect_480",   "sym8", 6, "garrote"],
    ["ph",            "db4",  4, "soft"]
], columns=["Sensor", "Wavelet", "Levels", "Threshold Mode"])

os.makedirs("summary_reports", exist_ok=True)
sensor_config_table.to_csv("summary_reports/sensor_config_formal.csv", index=False)

# Convert to config dict
sensor_config = {
    row["Sensor"]: {
        "wavelet": row["Wavelet"],
        "level": row["Levels"],
        "threshold_multiplier": 0.8 if "reflect" in row["Sensor"] else 0.6 if "temp" in row["Sensor"] else 0.7 if row["Sensor"] == "humidity" else 1.0,
        "mode": row["Threshold Mode"]
    }
    for _, row in sensor_config_table.iterrows()
}
sensor_config["default"] = {"wavelet": "db4", "level": 4, "threshold_multiplier": 1.0, "mode": "soft"}

sampling_rate = 1 / 300  # 1 sample per 5 minutes

# Create output folders
output_folders = [
    "wavelet_coeffs_csv",
    "wavelet_reconstructed_plots",
    "wavelet_wavedec_plots",
    "wavelet_freq_components_csv",
    "wavelet_freq_components_combined_csv",
    "wavelet_metrics",
    "summary_reports",
    "full_band_visualizations"
]
for folder in output_folders:
    os.makedirs(folder, exist_ok=True)

# Thresholding

def adaptive_threshold(detail_coeffs, multiplier=1.0, mode='soft'):
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745
    base_thresh = sigma * np.sqrt(2 * np.log(len(detail_coeffs)))
    coeff_std = np.std(detail_coeffs)
    coeff_range = np.ptp(detail_coeffs)
    dynamic_multiplier = multiplier * (1 + 0.2 * np.log1p(coeff_std / coeff_range))
    thresh = dynamic_multiplier * base_thresh

    if mode == 'soft':
        return pywt.threshold(detail_coeffs, thresh, 'soft')
    elif mode == 'hard':
        return pywt.threshold(detail_coeffs, thresh, 'hard')
    elif mode == 'garrote':
        garrote_coeffs = detail_coeffs.copy()
        mask = np.abs(detail_coeffs) > thresh
        garrote_coeffs[mask] = detail_coeffs[mask] - thresh**2 / detail_coeffs[mask]
        garrote_coeffs[~mask] = 0
        return garrote_coeffs
    return detail_coeffs

# Denoising function

def denoise_signal(signal, sensor_name):
    config = sensor_config.get(sensor_name, sensor_config["default"])
    wavelet_name = config["wavelet"]
    level = config["level"]
    multiplier = config["threshold_multiplier"]
    mode = config["mode"]

    wavelet = pywt.Wavelet(wavelet_name)
    max_level = pywt.dwt_max_level(len(signal), wavelet.dec_len)
    if level > max_level:
        level = max_level

    coeffs = pywt.wavedec(signal, wavelet=wavelet_name, level=level)
    coeffs_denoised = coeffs.copy()
    for i in range(1, len(coeffs_denoised)):
        coeffs_denoised[i] = adaptive_threshold(coeffs_denoised[i], multiplier, mode)

    reconstructed = pywt.waverec(coeffs_denoised, wavelet_name)
    return reconstructed[:len(signal)], coeffs, coeffs_denoised

# Processing

metrics = []
cleaned_signals = {}

for col in sensor_columns:
    signal = df[col].ffill().values
    if np.var(signal) < 1e-10:
        cleaned_signals[col] = signal
        continue

    reconstructed, coeffs, denoised_coeffs = denoise_signal(signal, col)
    cleaned_signals[col] = reconstructed

    # Save wavelet coeffs
    for i, c in enumerate(coeffs):
        pd.DataFrame({f"{col}_L{i}": c}).to_csv(f"wavelet_coeffs_csv/{col}_L{i}.csv", index=False)

    # Save raw wavelet coefficients (padded) into one CSV per sensor
    level = sensor_config.get(col, sensor_config["default"])["level"]
    coeff_names = [f"cA_{level}"] + [f"cD_{level - i}" for i in range(level)]
    max_len = max(len(c) for c in coeffs)
    padded_coeffs = [np.pad(c, (0, max_len - len(c)), mode='constant') for c in coeffs]
    coeff_df = pd.DataFrame({name: coeff for name, coeff in zip(coeff_names, padded_coeffs)})
    coeff_df.to_csv(f"wavelet_freq_components_combined_csv/{col}_bands.csv", index=False)

    # Compute metrics
    try:
        mse = mean_squared_error(signal, reconstructed)
        rmse = np.sqrt(mse)
        power = np.mean(signal ** 2)
        snr = 10 * np.log10(power / mse) if mse > 0 else 0
        ssim_val = 1.0 if signal.max() == signal.min() else ssim(signal, reconstructed, data_range=signal.max() - signal.min())
        metrics.append({"sensor": col, "wavelet": sensor_config[col]["wavelet"], "level": level, "threshold_mode": sensor_config[col]["mode"],
                        "MSE": mse, "RMSE": rmse, "SNR_dB": snr, "SSIM": ssim_val})
    except Exception as e:
        print(f"[!] Metric error for {col}: {e}")

    # Plot Original vs Denoised vs Residual
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1); plt.plot(signal); plt.title(f"{col} - Original"); plt.grid(True)
    plt.subplot(3, 1, 2); plt.plot(reconstructed); plt.title(f"{col} - Denoised"); plt.grid(True)
    plt.subplot(3, 1, 3); plt.plot(signal - reconstructed); plt.title(f"{col} - Residual"); plt.grid(True)
    plt.tight_layout(); plt.savefig(f"wavelet_reconstructed_plots/{col}_comparison.png"); plt.close()

# Save cleaned dataset
df_cleaned = pd.DataFrame(cleaned_signals)
df_cleaned["timestamp"] = df["timestamp"]
df_cleaned["growth_stage"] = df["growth_stage"]
df_cleaned["yield_count"] = df["yield_count"]
df_cleaned["batch_id"] = df["batch_id"]
df_cleaned.to_csv("lettuce_wavelet_cleaned.csv", index=False)

# Save metrics
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("wavelet_metrics/wavelet_denoising_metrics.csv", index=False)

# Save PDF Report
with PdfPages('summary_reports/wavelet_analysis_summary.pdf') as pdf:
    plt.figure(figsize=(14, 6)); plt.axis('off'); plt.title("Wavelet Denoising Metrics")
    table = plt.table(cellText=metrics_df.round(4).values, colLabels=metrics_df.columns,
                      cellLoc='center', loc='center'); table.scale(1.2, 1.2); pdf.savefig(); plt.close()

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    sns.barplot(x='sensor', y='MSE', data=metrics_df, ax=axs[0, 0]); axs[0, 0].set_title('MSE'); axs[0, 0].tick_params(axis='x', rotation=45)
    sns.barplot(x='sensor', y='RMSE', data=metrics_df, ax=axs[0, 1]); axs[0, 1].set_title('RMSE'); axs[0, 1].tick_params(axis='x', rotation=45)
    sns.barplot(x='sensor', y='SNR_dB', data=metrics_df, ax=axs[1, 0]); axs[1, 0].set_title('SNR (dB)'); axs[1, 0].tick_params(axis='x', rotation=45)
    sns.barplot(x='sensor', y='SSIM', data=metrics_df, ax=axs[1, 1]); axs[1, 1].set_title('SSIM'); axs[1, 1].tick_params(axis='x', rotation=45)
    plt.tight_layout(); pdf.savefig(); plt.close()

print("✅ Denoising complete. Raw wavelet coefficients saved in 'wavelet_freq_components_combined_csv/'.")
