# wavelet_processing.py
import pandas as pd
import numpy as np
import pywt
import os
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("lettuce_growth_final_with_yieldsu.csv", parse_dates=["timestamp"])

# Sensor columns to denoise
sensor_columns = [
    "humidity", "temp_envi", "temp_water", "tds", "lux",
    "ppfd", "reflect_445", "reflect_480", "ph"
]

# Prepare folders
os.makedirs("wavelet_coeffs_csv", exist_ok=True)
os.makedirs("wavelet_reconstructed_plots", exist_ok=True)
os.makedirs("wavelet_wavedec_plots", exist_ok=True)
os.makedirs("wavelet_freq_components_csv", exist_ok=True)

# Sampling frequency (5 minutes interval → 1 sample every 300s)
sampling_rate = 1 / 300  # Hz
wavelet_name = 'db4'
level = 4
wavelet = pywt.Wavelet(wavelet_name)

# Container for cleaned signals
cleaned_signals = {}

# Apply wavelet decomposition and reconstruction
for col in sensor_columns:
    signal = df[col].ffill().values

    # Wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet=wavelet_name, level=level)

    # Save coefficients
    for i, c in enumerate(coeffs):
        pd.DataFrame({f"{col}_L{i}": c}).to_csv(f"wavelet_coeffs_csv/{col}_L{i}.csv", index=False)

    # Zero detail coefficients for denoising
    coeffs_denoised = coeffs.copy()
    for i in range(1, len(coeffs_denoised)):
        coeffs_denoised[i] = np.zeros_like(coeffs_denoised[i])

    reconstructed = pywt.waverec(coeffs_denoised, wavelet_name)
    reconstructed = reconstructed[:len(signal)]
    cleaned_signals[col] = reconstructed

    # Plot original vs. denoised
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

    # Plot each frequency component
    fig, axs = plt.subplots(len(coeffs)+1, 1, figsize=(15, 3 * (len(coeffs)+1)), sharex=True)
    axs[0].plot(signal, color="black")
    axs[0].set_title(f"{col} - Original Signal")
    axs[0].grid(True)

    for i in range(len(coeffs)):
        coeffs_band = [np.zeros_like(c) for c in coeffs]
        coeffs_band[i] = coeffs[i]
        band_signal = pywt.waverec(coeffs_band, wavelet_name)[:len(signal)]

        band_type = f"cA_{level}" if i == 0 else f"cD_{level - i + 1}"
        axs[i+1].plot(band_signal, label=band_type)
        axs[i+1].set_title(f"{col} - {band_type} (Approx. Frequency Component)")
        axs[i+1].grid(True)

        scale = 2 ** (level - i)
        freq = pywt.scale2frequency(wavelet, scale) * sampling_rate
        axs[i+1].legend(title=f"~ {freq:.6f} Hz")

        pd.DataFrame({f"{col}_{band_type}": band_signal}).to_csv(
            f"wavelet_freq_components_csv/{col}_{band_type}.csv", index=False
        )

    fig.tight_layout()
    plt.savefig(f"wavelet_wavedec_plots/{col}_wavedec_bands.png")
    plt.close()
    print(f"✅ Saved frequency bands: wavelet_wavedec_plots/{col}_wavedec_bands.png")

# Create final DataFrame
df_cleaned = pd.DataFrame(cleaned_signals)
df_cleaned["timestamp"]     = df["timestamp"]
df_cleaned["growth_stage"]  = df["growth_stage"]
df_cleaned["yield_count"]   = df["yield_count"]
df_cleaned["batch_id"]      = df["batch_id"]    # ← restore original batch_id

# Save cleaned CSV
df_cleaned.to_csv("lettuce_wavelet_cleaned.csv", index=False)
print("✅ Wavelet denoised data saved to lettuce_wavelet_cleaned.csv")
