import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os

# 1. Load dataset
df = pd.read_csv("lettuce_growth_finalized.csv", parse_dates=["timestamp"])

# 2. Sensor columns
sensor_columns = [
    "humidity", "temp_envi", "temp_water", "tds", "lux",
    "ppfd", "reflect_445", "reflect_480", "ph"
]

# 3. Prepare folders
os.makedirs("wavelet_denoised_plots", exist_ok=True)
os.makedirs("wavelet_coeffs_csv", exist_ok=True)

# 4. Dictionary for cleaned signals
cleaned_signals = {}

# 5. Wavelet Denoising for each sensor column
for col in sensor_columns:
    signal = df[col].values

    # Fill NaNs if present
    if np.any(pd.isna(signal)):
        print(f"Warning: {col} contains NaNs. Filling forward.")
        signal = pd.Series(signal).fillna(method='ffill').values

    # Perform 1-level DWT using Daubechies-4
    cA, cD = pywt.dwt(signal, 'db4')

    # Save wavelet coefficients
    coeffs_df = pd.DataFrame({
        f"{col}_cA": cA,
        f"{col}_cD": cD
    })
    coeffs_df.to_csv(f"wavelet_coeffs_csv/{col}_wavelet_coeffs.csv", index=False)

    # Denoising: remove high-frequency components (zero detail)
    cD_clean = np.zeros_like(cD)
    cleaned_signal = pywt.idwt(cA, cD_clean, 'db4')
    cleaned_signal = cleaned_signal[:len(signal)]  # Match original length

    # Store for final cleaned dataset
    cleaned_signals[col] = cleaned_signal

    # Plot original vs denoised in two subplots
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axs[0].plot(signal, label="Original", color="blue", alpha=0.6)
    axs[0].set_title(f"{col} - Original Signal")
    axs[0].set_ylabel(col)
    axs[0].grid(True)

    axs[1].plot(cleaned_signal, label="Denoised", color="green", linewidth=1.2)
    axs[1].set_title(f"{col} - Denoised Signal (IDWT)")
    axs[1].set_xlabel("Sample Index")
    axs[1].set_ylabel(col)
    axs[1].grid(True)

    fig.tight_layout()
    plot_path = f"wavelet_denoised_plots/{col}_denoised.png"
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    plt.show()
    plt.close()

# 6. Save cleaned dataset
cleaned_df = pd.DataFrame(cleaned_signals)
cleaned_df["timestamp"] = df["timestamp"]
cleaned_df["growth_stage"] = df["growth_stage"]
cleaned_df.to_csv("lettuce_wavelet_cleaned.csv", index=False)
print("✅ Cleaned data saved to lettuce_wavelet_cleaned.csv")
