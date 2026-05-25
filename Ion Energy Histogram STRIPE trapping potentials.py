import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

plt.rcParams.update({
    'text.usetex': True,  # Enable LaTeX rendering
    'font.size': 18,      # Set global font size
    'font.family': 'serif'
})

def gaussian(x, amplitude, mean, stddev):
    """Define a Gaussian function for fitting."""
    return amplitude * np.exp(-((x - mean) / stddev)**2 / 2)

def calculate_fwhm(data, n_bootstraps=1000):
    """Calculate the FWHM directly from the histogram, with Poisson errors."""
    # Create a histogram of the data
    hist, bin_edges = np.histogram(data, bins=calculate_bins(data), density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Find the peak of the histogram
    peak_index = np.argmax(hist)
    peak_height = hist[peak_index]
    half_max = peak_height / 2

    # Find the indices where the histogram crosses half_max
    # Left side (from peak to lower bins)
    left_indices = np.where(hist[:peak_index] <= half_max)[0]
    left_index = left_indices[-1] if len(left_indices) > 0 else 0
    left_x = bin_centers[left_index]

    # Right side (from peak to higher bins)
    right_indices = np.where(hist[peak_index:] <= half_max)[0]
    right_index = peak_index + (right_indices[0] if len(right_indices) > 0 else 0)
    right_x = bin_centers[right_index]

    # Calculate FWHM
    fwhm = right_x - left_x

    # Calculate Poisson error on FWHM
    # Error on left_x and right_x due to Poisson statistics in the bins
    left_error = np.sqrt(hist[left_index]) * bin_width / peak_height
    right_error = np.sqrt(hist[right_index]) * bin_width / peak_height
    fwhm_error = np.sqrt(left_error**2 + right_error**2)

    # Bootstrap to estimate FWHM error
    fwhm_bootstraps = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=len(data), replace=True)
        hist_bs, bin_edges_bs = np.histogram(sample, bins=calculate_bins(data), density=False)
        bin_centers_bs = (bin_edges_bs[:-1] + bin_edges_bs[1:]) / 2
        bin_width_bs = bin_edges_bs[1] - bin_edges_bs[0]

        peak_index_bs = np.argmax(hist_bs)
        peak_height_bs = hist_bs[peak_index_bs]
        half_max_bs = peak_height_bs / 2

        left_indices_bs = np.where(hist_bs[:peak_index_bs] <= half_max_bs)[0]
        left_index_bs = left_indices_bs[-1] if len(left_indices_bs) > 0 else 0
        left_x_bs = bin_centers_bs[left_index_bs]

        right_indices_bs = np.where(hist_bs[peak_index_bs:] <= half_max_bs)[0]
        right_index_bs = peak_index_bs + (right_indices_bs[0] if len(right_indices_bs) > 0 else 0)
        right_x_bs = bin_centers_bs[right_index_bs]

        fwhm_bs = right_x_bs - left_x_bs
        fwhm_bootstraps.append(fwhm_bs)

    fwhm_error_bootstrap = np.std(fwhm_bootstraps) if fwhm_bootstraps else np.nan

    # Use the larger of the two errors (from Poisson or bootstrap)
    if not np.isnan(fwhm_error_bootstrap):
        fwhm_error = max(fwhm_error, fwhm_error_bootstrap)

    # Calculate mean and its Poisson error
    mean = np.sum(bin_centers * hist) / np.sum(hist)
    mean_error = np.std(data) / np.sqrt(len(data))

    return fwhm, fwhm_error, mean, mean_error

def calculate_bins(data):
    # Freedman-Diaconis rule
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    bin_width = 2 * iqr / (len(data) ** (1/3))
    num_bins = int((np.max(data) - np.min(data)) / bin_width)
    return np.linspace(np.min(data), np.max(data), num_bins + 1)

def load_data(file_path):
    """Load and process data from a single CSV file."""
    positions_data = pd.read_csv(file_path, skiprows=1, delimiter=',', header=None,
                                  names=["Pos_x", "Pos_y", "Pos_z", "Vel_x", "Vel_y", "Vel_z",
                                         "Energy_x", "Energy_y", "Energy_z", "ToF"])
    energy_z = positions_data["Energy_z"] * 10**3  # Convert to meV
    return energy_z

def plot_shifted_axial_energies_from_files(data_dir, file_names, labels):
    """Plot axial energy histograms with means shifted to align, including FWHM and mean errors in labels."""
    # Load all data and calculate means and SEM
    all_energy_z = []
    fwhms = []
    fwhm_errors = []
    means = []
    mean_errors = []
    max_spread = []

    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        energy_z = load_data(file_path)
        all_energy_z.extend(energy_z)

        # Calculate FWHM, its error, and mean with Poisson errors
        fwhm, fwhm_error, mean, mean_error = calculate_fwhm(energy_z)
        fwhms.append(fwhm)
        fwhm_errors.append(fwhm_error)
        means.append(mean)
        mean_errors.append(mean_error)

        # Calculate max spread
        max_spread.append(np.max(energy_z) - np.min(energy_z))

    # Calculate the overall mean of means (or choose a reference mean)
    reference_mean = 0

    # Shift all data to align means
    shifted_all_energy_z = []
    for i, file_name in enumerate(file_names):
        file_path = os.path.join(data_dir, file_name)
        energy_z = load_data(file_path)
        shifted_energy_z = energy_z - means[i] + reference_mean
        shifted_all_energy_z.extend(shifted_energy_z)

    # Define fixed bins based on shifted data
    bins = calculate_bins(shifted_all_energy_z)

    # Create the main plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)  # Main axes

    # Define colors and linestyles
    colors = ['tab:red', 'tab:blue', 'tab:purple', 'tab:orange', 'tab:brown', 'tab:green']
    linestyles = ['-', '-', '-', '-', '-', '-']

    # Plot histograms with shifted means
    for i, file_name in enumerate(file_names):
        file_path = os.path.join(data_dir, file_name)
        energy_z = load_data(file_path)
        shifted_energy_z = energy_z - means[i] + reference_mean

        label_with_data = (
            f"{labels[i]}\n"
            rf"$\langle E \rangle$: {means[i]:.2f} ± {mean_errors[i]:.2f} meV" + f"\n"
            f"FWHM: {fwhms[i]:.2f} ± {fwhm_errors[i]:.2f} meV"
        )

        ax.hist(
            shifted_energy_z,
            bins=bins,
            histtype='step',
            alpha=0.8,
            label=label_with_data,
            linewidth=2,
            color=colors[i],
            linestyle=linestyles[i]
        )

    ax.set_xlabel(r'$E - \langle E \rangle$ (meV)')
    ax.set_ylabel(r'Count')
    ax.set_title(r'\textbf{Extraction from different trapping potentials}')
    ax.legend(
        loc='upper right',
        borderaxespad=0.,
        fontsize='small',
        framealpha=1,
        edgecolor='black',
        fancybox=False,
    )
    ax.grid(True)

    plt.savefig(os.path.join(data_dir, 'comparing_diff_STRIPE_trapping_potentials_mean_shifted_energie_spread.png'), format='PNG', dpi=200,
                    bbox_inches='tight')
    plt.show()


script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "STRIPE-IG-BICEPS_data")
labels = [
    r"ES 6-7 ($z=0$ mm)",
    r"ES 6-7 ($z=50$ mm)",
    r"ES 6 ($z=0$ mm)",
    r"ES 6 ($z=50$ mm)"
]

# List of files to process (adjust as needed)
file_names = ['09042026/data 1/rawdata_IG_1.csv', '09042026/data 1/rawdata_IG_4.csv', '09042026/data 2/rawdata_IG_1.csv', '09042026/data 2/rawdata_IG_4.csv']

plot_shifted_axial_energies_from_files(data_dir, file_names, labels)