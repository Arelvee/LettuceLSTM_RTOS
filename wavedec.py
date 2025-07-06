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

# Sensor-specific configurations
sensor_config = {
    "reflect_445": {"wavelet": "sym8", "level": 6, "threshold_multiplier": 0.8, "mode": "garrote"},
    "reflect_480": {"wavelet": "sym8", "level": 6, "threshold_multiplier": 0.8, "mode": "garrote"},
    "temp_envi": {"wavelet": "db4", "level": 3, "threshold_multiplier": 0.6, "mode": "hard"},
    "temp_water": {"wavelet": "db4", "level": 3, "threshold_multiplier": 0.6, "mode": "hard"},
    "humidity": {"wavelet": "db8", "level": 5, "threshold_multiplier": 0.7, "mode": "soft"},
    # Default configuration for other sensors
    "default": {"wavelet": "db4", "level": 4, "threshold_multiplier": 1.0, "mode": "soft"}
}

sampling_rate = 1 / 300  # 1 sample per 5 mins

# Output folders
output_folders = [
    "wavelet_coeffs_csv",
    "wavelet_reconstructed_plots",
    "wavelet_wavedec_plots",
    "wavelet_freq_components_csv",
    "wavelet_metrics",
    "summary_reports",
    "full_band_visualizations"
]

for folder in output_folders:
    os.makedirs(folder, exist_ok=True)

# Enhanced thresholding with sensor-specific adjustments
def adaptive_threshold(detail_coeffs, multiplier=1.0, mode='soft'):
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745
    base_thresh = sigma * np.sqrt(2 * np.log(len(detail_coeffs)))
    
    # Adaptive multiplier based on coefficient distribution
    coeff_std = np.std(detail_coeffs)
    coeff_range = np.ptp(detail_coeffs)
    dynamic_multiplier = multiplier * (1 + 0.2 * np.log1p(coeff_std/coeff_range))
    
    thresh = dynamic_multiplier * base_thresh
    
    # Apply threshold with different modes
    if mode == 'soft':
        return pywt.threshold(detail_coeffs, thresh, 'soft')
    elif mode == 'hard':
        return pywt.threshold(detail_coeffs, thresh, 'hard')
    elif mode == 'garrote':
        # Garrote thresholding (less aggressive than soft)
        garrote_coeffs = detail_coeffs.copy()
        mask = np.abs(detail_coeffs) > thresh
        garrote_coeffs[mask] = detail_coeffs[mask] - thresh**2 / detail_coeffs[mask]
        garrote_coeffs[~mask] = 0
        return garrote_coeffs
    else:
        return pywt.threshold(detail_coeffs, thresh, 'soft')

# Enhanced denoising function
def denoise_signal(signal, sensor_name):
    config = sensor_config.get(sensor_name, sensor_config["default"])
    wavelet_name = config["wavelet"]
    level = config["level"]
    multiplier = config["threshold_multiplier"]
    mode = config["mode"]
    
    wavelet = pywt.Wavelet(wavelet_name)
    max_level = pywt.dwt_max_level(len(signal), wavelet.dec_len)
    
    # Adjust level if exceeds maximum possible
    if level > max_level:
        level = max_level
        print(f"[!] Adjusted level to {max_level} for {sensor_name}")
    
    coeffs = pywt.wavedec(signal, wavelet=wavelet_name, level=level)
    coeffs_denoised = coeffs.copy()
    
    # Preserve approximation coefficients
    for i in range(1, len(coeffs_denoised)):
        coeffs_denoised[i] = adaptive_threshold(
            coeffs_denoised[i], 
            multiplier=multiplier,
            mode=mode
        )
    
    reconstructed_signal = pywt.waverec(coeffs_denoised, wavelet_name)
    return reconstructed_signal[:len(signal)], coeffs, coeffs_denoised

# Process each sensor
metrics = []
cleaned_signals = {}
all_bands_data = {}

for col in sensor_columns:
    signal = df[col].ffill().values

    if np.var(signal) < 1e-10:
        print(f"[i] {col} has flat signal. Continuing with processing.")
        cleaned_signals[col] = signal
        continue

    reconstructed, coeffs, denoised_coeffs = denoise_signal(signal, col)
    cleaned_signals[col] = reconstructed

    # Save wavelet coefficients
    for i, c in enumerate(coeffs):
        pd.DataFrame({f"{col}_L{i}": c}).to_csv(f"wavelet_coeffs_csv/{col}_L{i}.csv", index=False)

    # Metrics
    try:
        mse_val = mean_squared_error(signal, reconstructed)
        rmse_val = np.sqrt(mse_val)
        signal_power = np.mean(signal ** 2)
        snr_val = 10 * np.log10(signal_power / mse_val) if mse_val > 0 else 0
        
        # Handle constant signals for SSIM
        if signal.max() == signal.min():
            ssim_val = 1.0
        else:
            ssim_val = ssim(signal, reconstructed, data_range=signal.max() - signal.min())

        metrics.append({
            "sensor": col,
            "wavelet": sensor_config.get(col, sensor_config["default"])["wavelet"],
            "level": sensor_config.get(col, sensor_config["default"])["level"],
            "threshold_mode": sensor_config.get(col, sensor_config["default"])["mode"],
            "MSE": mse_val,
            "RMSE": rmse_val,
            "SNR_dB": snr_val,
            "SSIM": ssim_val
        })

        print(f"Processed: {col} | MSE={mse_val:.4f}, RMSE={rmse_val:.4f}, SNR={snr_val:.2f} dB, SSIM={ssim_val:.4f}")
    except Exception as e:
        print(f"[!] Error computing metrics for {col}: {e}")

    # Enhanced plot: Original vs Denoised with Residual
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(signal, label="Original", alpha=0.7, color='blue')
    plt.title(f"{col} - Original Signal", fontsize=14)
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(reconstructed, label="Denoised", color='orange')
    plt.title(f"{col} - Reconstructed Signal", fontsize=14)
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    residual = signal - reconstructed
    plt.plot(residual, label="Residual", color='green')
    plt.title(f"{col} - Residual (Original - Denoised)", fontsize=14)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"wavelet_reconstructed_plots/{col}_enhanced_comparison.png", dpi=300)
    plt.close()

    # Frequency band decomposition
    wavelet_name = sensor_config.get(col, sensor_config["default"])["wavelet"]
    wavelet = pywt.Wavelet(wavelet_name)
    fig, axs = plt.subplots(len(coeffs) + 1, 1, figsize=(15, 3 * (len(coeffs) + 1)), sharex=True)
    fig.suptitle(f"{col} - Wavelet Frequency Bands ({wavelet_name})", fontsize=16)
    
    axs[0].plot(signal, color="black")
    axs[0].set_title("Original Signal", fontsize=12)
    axs[0].grid(True)

    band_signals = {}
    for i in range(len(coeffs)):
        coeffs_band = [np.zeros_like(c) for c in coeffs]
        coeffs_band[i] = denoised_coeffs[i]
        band_signal = pywt.waverec(coeffs_band, wavelet_name)[:len(signal)]
    
        # Get the actual level used (from config or adjusted)
        current_level = sensor_config.get(col, sensor_config["default"])["level"]
        wavelet = pywt.Wavelet(wavelet_name)
        max_level = pywt.dwt_max_level(len(signal), wavelet.dec_len)
        if current_level > max_level:
            current_level = max_level
    
        band_type = f"cA_{current_level}" if i == 0 else f"cD_{current_level - i + 1}"
        band_signals[band_type] = band_signal
        
        axs[i+1].plot(band_signal, label=band_type)
        axs[i+1].set_title(f"{band_type} Component", fontsize=12)
        axs[i+1].grid(True)
        
        try:
            if i == 0:
                band_level = level
            else:
                band_level = level - i + 1
            scale = 2 ** band_level
            freq = pywt.scale2frequency(wavelet, scale) * sampling_rate
            freq = max(freq, 1e-6)  # Ensure positive frequency
            axs[i+1].legend(title=f"~ {freq:.6f} Hz", fontsize=10)
        except:
            axs[i+1].legend(title="Frequency: N/A", fontsize=10)
        
        pd.DataFrame({f"{col}_{band_type}": band_signal}).to_csv(
            f"wavelet_freq_components_csv/{col}_{band_type}.csv", index=False
        )
    
    all_bands_data[col] = band_signals
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"wavelet_wavedec_plots/{col}_wavedec_bands.png", dpi=300)
    plt.close()

# Save cleaned dataset
df_cleaned = pd.DataFrame(cleaned_signals)
df_cleaned["timestamp"] = df["timestamp"]
df_cleaned["growth_stage"] = df["growth_stage"]
df_cleaned["yield_count"] = df["yield_count"]
df_cleaned["batch_id"] = df["batch_id"]
df_cleaned.to_csv("lettuce_wavelet_cleaned.csv", index=False)

# Enhanced metrics handling
metrics_df = pd.DataFrame(metrics).dropna()
metrics_df.to_csv("wavelet_metrics/wavelet_denoising_metrics.csv", index=False)

# Create comprehensive PDF report
with PdfPages('summary_reports/wavelet_analysis_summary.pdf') as pdf:
    # Metrics table
    plt.figure(figsize=(14, 6))
    plt.axis('off')
    plt.title("Wavelet Denoising Metrics with Configuration", fontsize=16)
    table = plt.table(
        cellText=metrics_df.round(4).values,
        colLabels=metrics_df.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.2)
    pdf.savefig()
    plt.close()

    # Metrics visualization
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Wavelet Denoising Performance Metrics', fontsize=20)
    
    sns.barplot(x='sensor', y='MSE', data=metrics_df, ax=axs[0, 0], palette='viridis')
    axs[0, 0].set_title('Mean Squared Error (MSE)', fontsize=14)
    axs[0, 0].tick_params(axis='x', rotation=45)
    
    sns.barplot(x='sensor', y='RMSE', data=metrics_df, ax=axs[0, 1], palette='viridis')
    axs[0, 1].set_title('Root Mean Squared Error (RMSE)', fontsize=14)
    axs[0, 1].tick_params(axis='x', rotation=45)
    
    sns.barplot(x='sensor', y='SNR_dB', data=metrics_df, ax=axs[1, 0], palette='viridis')
    axs[1, 0].set_title('Signal-to-Noise Ratio (SNR in dB)', fontsize=14)
    axs[1, 0].tick_params(axis='x', rotation=45)
    axs[1, 0].axhline(20, color='r', linestyle='--', alpha=0.7)  # SNR threshold
    
    sns.barplot(x='sensor', y='SSIM', data=metrics_df, ax=axs[1, 1], palette='viridis')
    axs[1, 1].set_title('Structural Similarity Index (SSIM)', fontsize=14)
    axs[1, 1].tick_params(axis='x', rotation=45)
    axs[1, 1].axhline(0.7, color='r', linestyle='--', alpha=0.7)  # SSIM threshold
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig()
    plt.close()

    # Full band visualization for each sensor
    for col in sensor_columns:
        if col not in all_bands_data:
            continue
            
        fig, ax = plt.subplots(figsize=(15, 8))
        config = sensor_config.get(col, sensor_config["default"])
        plt.title(f"{col} - Full Band Decomposition ({config['wavelet']}, Level={config['level']})", fontsize=18)
        
        # Plot original and reconstructed
        plt.plot(df.index, df[col], label='Original', alpha=0.7, linewidth=2)
        plt.plot(df.index, cleaned_signals[col], label='Denoised', linewidth=1.5)
        
        # Plot all bands
        for band, signal in all_bands_data[col].items():
            plt.plot(df.index, signal, '--', label=f"{band} Band", alpha=0.8)
        
        plt.xlabel('Time Index', fontsize=12)
        plt.ylabel('Sensor Value', fontsize=12)
        plt.grid(True)
        plt.legend(fontsize=10, loc='upper right')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

print("Processing complete with sensor-specific optimizations.")
print("Results saved in organized folders.")