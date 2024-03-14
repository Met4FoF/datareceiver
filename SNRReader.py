import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import json

def read_data(folder_path):
    file_pattern = f"{folder_path}/MultiSineFFTComp_*_Exp_*_Axis_*_freq_*_HzSNR_params.json"
    data = []
    # Read each JSON file and append its data to the list
    for file_path in glob.glob(file_pattern):
        with open(file_path, 'r') as file:
            json_data = json.load(file)

            # Flatten the FFTSNR dictionary
            for key, value in json_data['FFTSNR'].items():
                for subkey, subvalue in value.items():
                    # Create a new key like '2_linear'
                    new_key = f"{key}_{subkey}"
                    json_data[new_key] = subvalue

            # Remove the original FFTSNR dictionary
            del json_data['FFTSNR']

            # Assign sineSNR an interpolation factor of one
            json_data['1_sineSNR'] = json_data['sineSNR']
            del json_data['sineSNR']

            data.append(json_data)
    df = pd.DataFrame(data)
    return df

def normalize_snr(df):
    snr_columns = [col for col in df.columns if 'linear' in col or col == '1_sineSNR']
    for col in snr_columns:
        df[col] = df[col] / df['1_sineSNR']
    return df

def bin_data(df, step=0.1):
    min_freq = df['freq'].min()
    max_freq = df['freq'].max()
    bins = [round(b, 1) for b in np.arange(min_freq, max_freq + step, step)]
    df['freq_bin'] = df['freq'].apply(lambda x: min(bins, key=lambda b: abs(b - x)))
    return df

def calculate_stats(df):
    grouped = df.groupby('freq_bin').agg(['mean', 'std'])
    return grouped

def plot_data(grouped):
    fig, ax = plt.subplots(figsize=(10, 6))
    snr_columns = [col for col in grouped.columns.levels[0] if 'linear' in col]
    num_cols = len(snr_columns)
    offset_increment = 0.1  # Increased offset for each interpolation factor
    base_offset = -(offset_increment * num_cols) / 2  # Starting offset

    for i, col in enumerate(snr_columns):
        offset = base_offset + i * offset_increment
        # Adjust the x-axis values by the offset
        x_values = grouped.index + offset
        ax.errorbar(x_values, grouped[col]['mean'], yerr=grouped[col]['std'], label=col, fmt='o-', capsize=5)

    ax.set_xlabel('Frequency Bin (Hz)')
    ax.set_ylabel('Normalized SNR')
    ax.set_title('Normalized SNR Mean and Std by Frequency Bin')
    ax.legend()
    ax.set_xlim(right=160)  # Limit x-axis to 160 Hz
    ax.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
    plt.show()

def plot_sorted_snr_lines(grouped, raw_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    snr_columns = [col for col in grouped.columns.levels[0] if 'linear' in col]
    num_cols = len(snr_columns)
    offset_increment = 0.1  # Spacing between different interpolation factors
    half_width = (num_cols - 1) * offset_increment / 2

    # Store colors for each interpolation factor
    colors = []

    # Plotting mean values and storing colors
    for i, col in enumerate(snr_columns):
        offset = i * offset_increment - half_width
        x_values = grouped.index + offset
        line, = ax.plot(x_values, grouped[col]['mean'], 'o-', label=col, markersize=5)
        colors.append(line.get_color())

    # Plotting sorted SNR lines with matched colors
    for i, col in enumerate(snr_columns):
        offset = i * offset_increment - half_width
        color = colors[i]  # Use the stored color for the current interpolation factor

        for freq_bin in grouped.index:
            sorted_values = sorted(raw_df[raw_df['freq_bin'] == freq_bin][col])
            num_lines = len(sorted_values) // 2
            line_alpha = 1 / max(num_lines, 1)

            for j in range(num_lines):
                x_val = freq_bin + offset
                # Plot vertical lines with reduced alpha
                ax.plot([x_val, x_val], [sorted_values[j], sorted_values[-(j+1)]], color=color, alpha=line_alpha, linewidth=3)
                # Plot markers with alpha 1
                ax.plot([x_val, x_val], [sorted_values[j], sorted_values[-(j+1)]], color=color, alpha=1, linewidth=0, marker='_')

    ax.set_xlabel('Frequency Bin (Hz)')
    ax.set_ylabel('Normalized SNR')
    ax.set_title('Normalized SNR Mean and Sorted Range by Frequency Bin')
    ax.legend()
    ax.set_xlim(0, 160)  # Limit x-axis to 160 Hz
    ax.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
    plt.show()

# Replace 'plot_data(grouped)' with 'plot_sorted_snr_lines(grouped, df)' in your main function


def main():
    folder_path = 'SNRParams'
    df = read_data(folder_path)
    df = normalize_snr(df)
    df = bin_data(df)
    grouped = calculate_stats(df)
    plot_sorted_snr_lines(grouped,df)

if __name__ == "__main__":
    main()