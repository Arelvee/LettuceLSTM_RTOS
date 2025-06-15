import os
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("lettuce_growth_final_with_yieldsu.csv", parse_dates=["timestamp"])

# Columns to denoise
sensor_columns = [
    "humidity", "temp_envi", "temp_water", "tds", "lux",
    "ppfd", "reflect_445", "reflect_480", "ph"
]

# Setup
wavelet_name = 'db4'
level = 4
sampling_rate = 1 / 300  # 1 sample per 5 mins
wavelet = pywt.Wavelet(wavelet_name)

# Output folders
os.makedirs("wavelet_coeffs_csv", exist_ok=True)
os.makedirs("wavelet_reconstructed_plots", exist_ok=True)
os.makedirs("wavelet_wavedec_plots", exist_ok=True)
os.makedirs("wavelet_freq_components_csv", exist_ok=True)
os.makedirs("wavelet_metrics", exist_ok=True)

# Metric container
metrics = []
cleaned_signals = {}

# Universal threshold (VisuShrink)
def universal_threshold(detail_coeffs):
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745
    return sigma * np.sqrt(2 * np.log(len(detail_coeffs)))

# Process each sensor column
for col in sensor_columns:
    signal = df[col].ffill().values
    coeffs = pywt.wavedec(signal, wavelet=wavelet_name, level=level)

    # Save decomposition coefficients
    for i, c in enumerate(coeffs):
        pd.DataFrame({f"{col}_L{i}": c}).to_csv(f"wavelet_coeffs_csv/{col}_L{i}.csv", index=False)

    # Soft-threshold denoising
    coeffs_denoised = coeffs.copy()
    for i in range(1, len(coeffs_denoised)):
        thresh = universal_threshold(coeffs_denoised[i])
        coeffs_denoised[i] = pywt.threshold(coeffs_denoised[i], thresh, mode='soft')

    reconstructed = pywt.waverec(coeffs_denoised, wavelet_name)[:len(signal)]
    cleaned_signals[col] = reconstructed

    # Compute metrics
    try:
        mse_val = mean_squared_error(signal, reconstructed)
        rmse_val = np.sqrt(mse_val)
        signal_power = np.mean(signal ** 2)
        snr_val = 10 * np.log10(signal_power / mse_val) if mse_val > 0 else np.inf
        ssim_val = ssim(signal, reconstructed, data_range=signal.max() - signal.min())
        metrics.append({
            "sensor": col, "MSE": mse_val, "RMSE": rmse_val,
            "SNR_dB": snr_val, "SSIM": ssim_val
        })
        print(f"Processed: {col} | MSE={mse_val:.4f}, RMSE={rmse_val:.4f}, SNR={snr_val:.2f} dB, SSIM={ssim_val:.4f}")
    except Exception as e:
        print(f"[!] Skipped {col} metrics due to error: {e}")

    # Plot original vs denoised
    plt.figure(figsize=(12, 5))
    plt.subplot(2, 1, 1)
    plt.plot(signal, label="Original", alpha=0.7)
    plt.title(f"{col} - Original")
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(reconstructed, label="Denoised", color="orange")
    plt.title(f"{col} - Reconstructed")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"wavelet_reconstructed_plots/{col}_wavelet_plot.png")
    plt.close()

    # Frequency components
    fig, axs = plt.subplots(len(coeffs) + 1, 1, figsize=(15, 3 * (len(coeffs) + 1)), sharex=True)
    axs[0].plot(signal, color="black")
    axs[0].set_title(f"{col} - Original Signal")
    axs[0].grid(True)

    for i in range(len(coeffs)):
        coeffs_band = [np.zeros_like(c) for c in coeffs]
        coeffs_band[i] = coeffs[i]
        band_signal = pywt.waverec(coeffs_band, wavelet_name)[:len(signal)]
        band_type = f"cA_{level}" if i == 0 else f"cD_{level - i + 1}"
        axs[i + 1].plot(band_signal, label=band_type)
        axs[i + 1].set_title(f"{col} - {band_type}")
        axs[i + 1].grid(True)
        freq = pywt.scale2frequency(wavelet, 2 ** (level - i)) * sampling_rate
        axs[i + 1].legend(title=f"~ {freq:.6f} Hz")
        pd.DataFrame({f"{col}_{band_type}": band_signal}).to_csv(
            f"wavelet_freq_components_csv/{col}_{band_type}.csv", index=False)

    fig.tight_layout()
    plt.savefig(f"wavelet_wavedec_plots/{col}_wavedec_bands.png")
    plt.close()

# Save cleaned data
df_cleaned = pd.DataFrame(cleaned_signals)
df_cleaned["timestamp"] = df["timestamp"]
df_cleaned["growth_stage"] = df["growth_stage"]
df_cleaned["yield_count"] = df["yield_count"]
df_cleaned["batch_id"] = df["batch_id"]
df_cleaned.to_csv("lettuce_wavelet_cleaned.csv", index=False)
print("✅ Wavelet-denoised data saved to lettuce_wavelet_cleaned.csv")

# Save and plot metrics
metrics_df = pd.DataFrame(metrics).dropna()
metrics_df.to_csv("wavelet_metrics/wavelet_denoising_metrics.csv", index=False)

# Plot combined metrics
plt.figure(figsize=(12, 6))
for metric in ["MSE", "RMSE", "SNR_dB", "SSIM"]:
    plt.plot(metrics_df["sensor"], metrics_df[metric], marker='o', label=metric)
plt.title("Wavelet Denoising Metrics per Sensor")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("wavelet_metrics/wavelet_denoising_metrics_plot.png")
plt.close()

print("📊 Metrics and visualizations saved to 'wavelet_metrics/'")
