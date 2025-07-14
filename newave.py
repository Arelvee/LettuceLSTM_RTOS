import os
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import xlsxwriter

# Load dataset
df = pd.read_csv("lettuce_growth_data.csv", parse_dates=["timestamp"])

sensor_columns = [
    "humidity", "temp_envi", "temp_water", "tds", "ec",
    "lux", "ppfd", "reflect_445", "reflect_480", "ph"
]

# Enhanced sensor-specific configurations
sensor_config = {
    "reflect_445": {"wavelet": "sym8", "level": 6, "threshold_multiplier": 0.8, "mode": "garrote"},
    "reflect_480": {"wavelet": "sym8", "level": 6, "threshold_multiplier": 0.8, "mode": "garrote"},
    "temp_envi": {"wavelet": "db4", "level": 3, "threshold_multiplier": 0.6, "mode": "hard"},
    "temp_water": {"wavelet": "db4", "level": 3, "threshold_multiplier": 0.6, "mode": "hard"},
    "humidity": {"wavelet": "db8", "level": 5, "threshold_multiplier": 0.7, "mode": "soft"},
    "tds": {"wavelet": "coif3", "level": 4, "threshold_multiplier": 0.9, "mode": "soft"},
    "ec": {"wavelet": "coif3", "level": 4, "threshold_multiplier": 0.9, "mode": "soft"},
    "lux": {"wavelet": "sym6", "level": 5, "threshold_multiplier": 1.0, "mode": "garrote"},
    "ppfd": {"wavelet": "sym6", "level": 5, "threshold_multiplier": 1.0, "mode": "garrote"},
    "ph": {"wavelet": "db6", "level": 4, "threshold_multiplier": 0.8, "mode": "soft"},
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
    "full_band_visualizations",
    "temporal_analysis",
    "reconstruction_tables",
    "wavelet_coeffs_excel"  # New folder for Excel files
]

for folder in output_folders:
    os.makedirs(folder, exist_ok=True)

# Enhanced thresholding with statistical validation
def adaptive_threshold(detail_coeffs, multiplier=1.0, mode='soft'):
    # Handle constant signals
    if np.ptp(detail_coeffs) < 1e-10:
        return np.zeros_like(detail_coeffs)
    
    # Robust noise estimation
    sigma = stats.median_abs_deviation(detail_coeffs, scale='normal')
    if sigma < 1e-10:
        sigma = np.std(detail_coeffs)
    
    base_thresh = sigma * np.sqrt(2 * np.log(len(detail_coeffs)))
    
    # Dynamic multiplier based on kurtosis
    k = stats.kurtosis(detail_coeffs, fisher=False)
    dynamic_multiplier = multiplier * (1 + 0.1 * (k - 3))  # Adjust based on deviation from normal distribution
    
    thresh = max(1e-10, dynamic_multiplier * base_thresh)  # Ensure threshold > 0
    
    # Apply threshold with different modes
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
    else:
        return pywt.threshold(detail_coeffs, thresh, 'soft')

# Function to save coefficients to Excel with multiple sheets
def save_coeffs_to_excel(coeffs, denoised_coeffs, sensor_name, level):
    excel_path = f"wavelet_coeffs_excel/{sensor_name}_wavelet_coeffs.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        # Save original coefficients
        for i, c in enumerate(coeffs):
            band_type = "cA" if i == 0 else f"cD{level - i + 1}"
            pd.DataFrame({f"{band_type}": c}).to_excel(
                writer, sheet_name=f"Original_{band_type}", index=False
            )
        
        # Save denoised coefficients
        for i, c in enumerate(denoised_coeffs):
            band_type = "cA" if i == 0 else f"cD{level - i + 1}"
            pd.DataFrame({f"{band_type}": c}).to_excel(
                writer, sheet_name=f"Denoised_{band_type}", index=False
            )
        
        # Save combined coefficients in one sheet
        combined_data = {}
        for i, (orig, den) in enumerate(zip(coeffs, denoised_coeffs)):
            band_type = "cA" if i == 0 else f"cD{level - i + 1}"
            combined_data[f"{band_type}_orig"] = orig
            combined_data[f"{band_type}_den"] = den
            
            # Pad shorter arrays to match length
            max_len = max(len(orig), len(den))
            if len(orig) < max_len:
                combined_data[f"{band_type}_orig"] = np.pad(orig, (0, max_len - len(orig)), 'constant')
            if len(den) < max_len:
                combined_data[f"{band_type}_den"] = np.pad(den, (0, max_len - len(den)), 'constant')
        
        pd.DataFrame(combined_data).to_excel(
            writer, sheet_name="Combined_Coeffs", index=False
        )
    
    print(f"Saved Excel coefficients for {sensor_name}")

# Function to create reconstruction table
def create_reconstruction_table(coeffs_denoised, wavelet_name, signal_length):
    recon_steps = []
    recon_signals = []
    
    # Start with approximation coefficients
    approx_coeffs = [coeffs_denoised[0]] + [None] * (len(coeffs_denoised) - 1)
    recon = pywt.waverec(approx_coeffs, wavelet_name)[:signal_length]
    recon_steps.append(["Approximation Only", "cA"])
    recon_signals.append(recon)
    
    # Add detail coefficients one by one
    for i in range(1, len(coeffs_denoised)):
        current_coeffs = coeffs_denoised.copy()
        # Set all higher detail coefficients to zero
        for j in range(i+1, len(coeffs_denoised)):
            current_coeffs[j] = np.zeros_like(coeffs_denoised[j])
        
        recon = pywt.waverec(current_coeffs, wavelet_name)[:signal_length]
        bands = ["cA"] + [f"cD{len(coeffs_denoised)-j}" for j in range(1, i+1)]
        recon_steps.append([f"Level {len(coeffs_denoised)-i} Reconstruction", "+".join(bands)])
        recon_signals.append(recon)
    
    # Create DataFrame
    df_recon = pd.DataFrame({
        "Reconstruction Step": [step[0] for step in recon_steps],
        "Components": [step[1] for step in recon_steps]
    })
    
    # Add signal columns
    for i, signal in enumerate(recon_signals):
        df_recon[f"Step_{i}_Signal"] = np.pad(signal, (0, signal_length - len(signal)), 'constant')
    
    return df_recon

# Enhanced denoising function with signal validation
def denoise_signal(signal, sensor_name):
    # Validate signal
    if len(signal) < 10:
        print(f"[!] {sensor_name}: Signal too short ({len(signal)} samples). Returning original.")
        return signal, [], []
    
    if np.var(signal) < 1e-10:
        print(f"[i] {sensor_name}: Constant signal. Returning original.")
        return signal, [], []
    
    config = sensor_config.get(sensor_name, sensor_config["default"])
    wavelet_name = config["wavelet"]
    level = config["level"]
    multiplier = config["threshold_multiplier"]
    mode = config["mode"]
    
    # Calculate maximum decomposition level
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet_name).dec_len)
    if level > max_level:
        level = max_level
        print(f"[!] {sensor_name}: Adjusted level to {max_level}")
    
    try:
        coeffs = pywt.wavedec(signal, wavelet=wavelet_name, level=level)
    except ValueError as e:
        print(f"[!] {sensor_name}: Wavelet decomposition failed - {e}. Using original signal.")
        return signal, [], []
    
    coeffs_denoised = coeffs.copy()
    
    # Preserve approximation coefficients
    for i in range(1, len(coeffs_denoised)):
        coeffs_denoised[i] = adaptive_threshold(
            coeffs_denoised[i], 
            multiplier=multiplier,
            mode=mode
        )
    
    try:
        reconstructed_signal = pywt.waverec(coeffs_denoised, wavelet_name)
        # Handle potential length mismatch
        if len(reconstructed_signal) > len(signal):
            reconstructed_signal = reconstructed_signal[:len(signal)]
        elif len(reconstructed_signal) < len(signal):
            reconstructed_signal = np.pad(reconstructed_signal, (0, len(signal) - len(reconstructed_signal)), 'edge')
        
        # Save coefficients to Excel
        save_coeffs_to_excel(coeffs, coeffs_denoised, sensor_name, level)
        
        # Create reconstruction table
        recon_table = create_reconstruction_table(coeffs_denoised, wavelet_name, len(signal))
        recon_table.to_csv(f"reconstruction_tables/{sensor_name}_reconstruction_steps.csv", index=False)
        recon_table.to_excel(f"reconstruction_tables/{sensor_name}_reconstruction_steps.xlsx", index=False)
        
        return reconstructed_signal, coeffs, coeffs_denoised
    except Exception as e:
        print(f"[!] {sensor_name}: Reconstruction failed - {e}. Using original signal.")
        return signal, coeffs, coeffs_denoised

# Process each sensor
metrics = []
cleaned_signals = {}
all_bands_data = {}
temporal_features = []

for col in sensor_columns:
    print(f"\n{'='*50}")
    print(f"Processing sensor: {col}")
    print(f"{'='*50}")
    
    signal = df[col].ffill().bfill().values  # Robust missing value handling

    # Handle constant signals
    if np.var(signal) < 1e-10:
        print(f"[i] {col} has constant signal. Skipping processing.")
        cleaned_signals[col] = signal
        continue

    reconstructed, coeffs, denoised_coeffs = denoise_signal(signal, col)
    cleaned_signals[col] = reconstructed

    # Skip processing if decomposition failed
    if not coeffs:
        continue

    # Save wavelet coefficients
    for i, c in enumerate(coeffs):
        pd.DataFrame({f"{col}_L{i}": c}).to_csv(f"wavelet_coeffs_csv/{col}_L{i}.csv", index=False)

    # Metrics calculation with validation
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
            "wavelet": config["wavelet"],
            "level": config["level"],
            "threshold_mode": config["mode"],
            "MSE": mse_val,
            "RMSE": rmse_val,
            "SNR_dB": snr_val,
            "SSIM": ssim_val
        })

        print(f"Performance: MSE={mse_val:.4f}, RMSE={rmse_val:.4f}, "
              f"SNR={snr_val:.2f} dB, SSIM={ssim_val:.4f}")
    except Exception as e:
        print(f"[!] Error computing metrics for {col}: {e}")

    # Enhanced plot: Original vs Denoised with Residual
    plt.figure(figsize=(15, 12))
    
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
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.title(f"{col} - Residual (Original - Denoised)", fontsize=14)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"wavelet_reconstructed_plots/{col}_comparison.png", dpi=300)
    plt.close()

    # Frequency band decomposition with temporal features
    fig, axs = plt.subplots(len(coeffs) + 1, 1, figsize=(15, 3 * (len(coeffs) + 1)), sharex=True)
    fig.suptitle(f"{col} - Wavelet Frequency Bands ({config['wavelet']})", fontsize=16)
    
    axs[0].plot(signal, color="black")
    axs[0].set_title("Original Signal", fontsize=12)
    axs[0].grid(True)

    band_signals = {}
    sensor_features = {"sensor": col}
    
    for i in range(len(coeffs)):
        coeffs_band = [np.zeros_like(c) for c in coeffs]
        coeffs_band[i] = denoised_coeffs[i]
        band_signal = pywt.waverec(coeffs_band, config['wavelet'])
        
        # Handle length mismatch
        if len(band_signal) > len(signal):
            band_signal = band_signal[:len(signal)]
        elif len(band_signal) < len(signal):
            band_signal = np.pad(band_signal, (0, len(signal) - len(band_signal)), 'edge')
        
        band_type = f"cA_{level}" if i == 0 else f"cD_{level - i + 1}"
        band_signals[band_type] = band_signal
        
        # Calculate band features
        sensor_features[f"{band_type}_mean"] = np.mean(band_signal)
        sensor_features[f"{band_type}_std"] = np.std(band_signal)
        sensor_features[f"{band_type}_energy"] = np.sum(band_signal**2)
        
        axs[i+1].plot(band_signal, label=band_type)
        axs[i+1].set_title(f"{band_type} Component", fontsize=12)
        axs[i+1].grid(True)
        
        # Calculate frequency information
        try:
            if i == 0:
                band_level = level
            else:
                band_level = level - i + 1
            scale = 2 ** band_level
            freq = pywt.scale2frequency(config['wavelet'], scale) * sampling_rate
            freq = max(freq, 1e-6)  # Ensure positive frequency
            axs[i+1].legend(title=f"~ {freq:.6f} Hz", fontsize=10)
        except:
            axs[i+1].legend(title="Frequency: N/A", fontsize=10)
        
        # Save frequency component
        pd.DataFrame({f"{col}_{band_type}": band_signal}).to_csv(
            f"wavelet_freq_components_csv/{col}_{band_type}.csv", index=False
        )
    
    temporal_features.append(sensor_features)
    all_bands_data[col] = band_signals
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"wavelet_wavedec_plots/{col}_frequency_bands.png", dpi=300)
    plt.close()

    # Temporal feature analysis
    plt.figure(figsize=(12, 8))
    for band, band_signal in band_signals.items():
        plt.plot(band_signal, label=band, alpha=0.7)
    plt.title(f"{col} - Temporal Feature Bands", fontsize=16)
    plt.xlabel("Time Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"temporal_analysis/{col}_temporal_features.png", dpi=300)
    plt.close()

# Save cleaned dataset
df_cleaned = pd.DataFrame(cleaned_signals)
df_cleaned["timestamp"] = df["timestamp"]
df_cleaned["growth_stage"] = df["growth_stage"]
df_cleaned["yield_count"] = df["yield_count"]
df_cleaned["batch_id"] = df["batch_id"]
df_cleaned.to_csv("lettuce_wavelet_cleaned.csv", index=False)

# Save temporal features
pd.DataFrame(temporal_features).to_csv("temporal_analysis/sensor_temporal_features.csv", index=False)

# Enhanced metrics handling
if metrics:
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("wavelet_metrics/wavelet_denoising_metrics.csv", index=False)
else:
    print("[!] No metrics generated. Creating empty metrics file.")
    pd.DataFrame().to_csv("wavelet_metrics/wavelet_denoising_metrics.csv")

# Create comprehensive PDF report
with PdfPages('summary_reports/wavelet_analysis_summary.pdf') as pdf:
    plt.figure(figsize=(10, 6))
    plt.axis('off')
    plt.title("Wavelet Denoising Report Summary", fontsize=16, pad=20)
    plt.figtext(0.5, 0.5, 
               f"Wavelet Analysis Report\n\n"
               f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
               f"Sensors Processed: {len(sensor_columns)}\n"
               f"Successful Denoising: {len(cleaned_signals)}/{len(sensor_columns)}",
               ha="center", va="center", fontsize=12)
    pdf.savefig()
    plt.close()

    if metrics:
        # Metrics table
        plt.figure(figsize=(14, 8))
        plt.axis('off')
        plt.title("Wavelet Denoising Metrics", fontsize=16, pad=20)
        table = plt.table(
            cellText=metrics_df.round(4).values,
            colLabels=metrics_df.columns,
            cellLoc='center',
            loc='center',
            colColours=['#f0f0f0']*len(metrics_df.columns)
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
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
        
        plt.plot(df.index, df[col].values, label='Original', alpha=0.7, linewidth=2)
        plt.plot(df.index, cleaned_signals[col], label='Denoised', linewidth=1.5)
        
        for band, signal_band in all_bands_data[col].items():
            plt.plot(df.index, signal_band, '--', label=f"{band} Band", alpha=0.8)
        
        plt.xlabel('Timestamp', fontsize=12)
        plt.ylabel('Sensor Value', fontsize=12)
        plt.grid(True)
        plt.legend(fontsize=10, loc='upper right')
        plt.tight_layout()
        pdf.savefig()
        plt.savefig(f"full_band_visualizations/{col}_full_bands.png", dpi=300)
        plt.close()

print("\nProcessing complete with enhanced features:")
print("- Combined CA/CD coefficients saved in Excel format")
print("- Reconstruction tables showing step-by-step signal reconstruction")
print("- Detailed temporal analysis of frequency bands")
print("Results saved in organized folders.")