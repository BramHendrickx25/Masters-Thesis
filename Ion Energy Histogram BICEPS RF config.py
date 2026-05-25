import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

plt.rcParams.update({
    'text.usetex': True,  # Enable LaTeX rendering
    'font.size': 24,      # Set global font size
    'font.family': 'serif'
})

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev)**2 / 2)

def calculate_fwhm(data, n_bootstraps=1000):
    print("original data =")
    print(data)
    bins = calculate_bins(data)
    print("bins =")
    print(bins)
    hist, bin_edges = np.histogram(data, bins=calculate_bins(data), density=False)
    print(hist)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mask = hist > 0
    hist = hist[mask]
    bin_centers = bin_centers[mask]
    hist_errors = np.sqrt(hist)
    print(hist)
    amplitude_guess = np.max(hist)
    mean_guess = np.mean(data)
    stddev_guess = np.std(data)

    try:
        popt, pcov = curve_fit(
            gaussian,
            bin_centers,
            hist,
            p0=[amplitude_guess, mean_guess, stddev_guess],
            sigma=hist_errors,
            absolute_sigma=True
        )
        amplitude, mean, stddev = popt
        fwhm = 2 * np.sqrt(2 * np.log(2)) * stddev

        # Analytical error propagation from fit
        stddev_error = np.sqrt(pcov[2, 2])
        fwhm_error_fit = 2 * np.sqrt(2 * np.log(2)) * stddev_error

    except RuntimeError:
        fwhm = np.nan
        fwhm_error_fit = np.nan

    return fwhm, fwhm_error_fit, popt

def calculate_fwhm_no_fit(data, n_bootstraps=1000):
    """Calculate FWHM for non-Gaussian peaks using interpolation."""
    hist, bin_edges = np.histogram(data, bins=calculate_bins(data), density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Find peak center (bin with max counts)
    peak_idx = np.argmax(hist)
    peak_center = bin_centers[peak_idx]
    peak_height = hist[peak_idx]

    # Find half-maximum
    half_max = peak_height / 2

    # Interpolate to find left and right half-max points
    from scipy.interpolate import interp1d
    interp_func = interp1d(bin_centers, hist, kind='linear', bounds_error=False, fill_value=0)

    # Find x-values where hist = half_max
    # We'll search in the left and right neighborhoods of the peak
    left_idx = np.max([0, peak_idx - 5])
    right_idx = np.min([len(bin_centers) - 1, peak_idx + 5])
    x_left = bin_centers[left_idx:peak_idx+1]
    x_right = bin_centers[peak_idx:right_idx+1]

    # Find left half-max
    try:
        left_half = np.min(x_left[interp_func(x_left) >= half_max])
    except ValueError:
        left_half = bin_centers[left_idx]

    # Find right half-max
    try:
        right_half = np.max(x_right[interp_func(x_right) >= half_max])
    except ValueError:
        right_half = bin_centers[right_idx]

    fwhm = right_half - left_half

    # Bootstrap to estimate FWHM error
    fwhm_bootstraps = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=len(data), replace=True)
        try:
            fwhm_bs = calculate_fwhm_no_fit(sample, n_bootstraps=0)[0]  # Recursive call (no bootstrapping)
            fwhm_bootstraps.append(fwhm_bs)
        except:
            continue
    fwhm_error = np.std(fwhm_bootstraps) if fwhm_bootstraps else np.nan

    return fwhm, fwhm_error, peak_center

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
    energy_r = (positions_data["Energy_x"] + positions_data["Energy_y"]) * 10**3 # Convert to meV
    return energy_z, energy_r

def plot_axial_energies_from_files(data_dir, file_names, labels):
    """Plot axial energy histograms with means shifted to align, including FWHM and mean errors in labels."""
    # Load all data and calculate means and SEM
    all_energy_z = []
    means = []
    sem_means = []
    fwhms = []
    fwhm_errors = []
    max_spread = []
    gaussian_params = []  # Store Gaussian parameters for each dataset

    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        energy_z, _ = load_data(file_path)
        all_energy_z.extend(energy_z)
        means.append(np.mean(energy_z))
        sem_means.append(np.std(energy_z, ddof=1) / np.sqrt(len(energy_z)))

        # Calculate FWHM and its error, and store Gaussian parameters
        fwhm, fwhm_error, popt = calculate_fwhm(energy_z)
        fwhms.append(fwhm)
        fwhm_errors.append(fwhm_error)
        gaussian_params.append(popt)  # Store the fitted parameters

        # Calculate max spread
        max_spread.append(np.max(energy_z) - np.min(energy_z))

    all_energy_z = []
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        energy_z, _ = load_data(file_path)  # Unpack energy_z
        all_energy_z.extend(energy_z.tolist())  # Flatten the Series into the list

    # Define fixed bins based on shifted data
    bins = calculate_bins(all_energy_z)

    # Create the main plot
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111)

    # Define colors and linestyles
    colors = ['tab:red', 'tab:blue', 'tab:purple', 'tab:orange']
    linestyles = ['-', '-', '-', '-']

    # Plot data sets
    for i in range(0, len(file_names)):
        file_path = os.path.join(data_dir, file_names[i])
        energy_z, _ = load_data(file_path)

        label_with_data = (
                f"{labels[i]}\n"
                rf"$\langle E \rangle$: {means[i]:.2f} $\pm$ {sem_means[i]:.2f} meV" + f"\n"
                rf"FWHM: {fwhms[i]:.2f} $\pm$ {fwhm_errors[i]:.2f} meV"
        )

        # Plot histogram (raw counts)
        hist, bin_edges = np.histogram(energy_z, bins=bins, density=False)

        ax1.hist(
            energy_z,
            bins=bins,
            histtype='step',
            alpha=0.8,
            label=label_with_data,
            linewidth=2,
            color=colors[i],
            linestyle=linestyles[i]
        )

        # Fit Gaussian to the histogram (density=True for fitting)
        hist_density, _ = np.histogram(energy_z, bins=bins, density=True)
        bin_centers_density = (bin_edges[:-1] + bin_edges[1:]) / 2
        try:
            popt, _ = curve_fit(gaussian, bin_centers_density, hist_density, p0=[np.max(hist_density), np.mean(energy_z), np.std(energy_z)])
            amplitude, mean, stddev = popt
        except RuntimeError:
            amplitude, mean, stddev = np.max(hist_density), np.mean(energy_z), np.std(energy_z)

        # Scale the Gaussian to match the histogram counts
        total_counts = len(energy_z)
        #x_fit = np.linspace(np.min(energy_z), np.max(energy_z), 1000)
        #y_fit_density = gaussian(x_fit, amplitude, mean, stddev)
        #y_fit_counts = y_fit_density * total_counts * bin_width  # Scale to counts

        # Plot Gaussian fit (scaled to counts)
        #ax1.plot(
        #    x_fit, y_fit_counts,
        #    color=colors[i],
        #    linestyle='--',
        #    linewidth=2,
        #    label=f"Gaussian Fit"
        #)

    # Set labels and title
    ax1.set_xlabel(r'$E$ (meV)')
    ax1.set_ylabel(r'Counts')
    ax1.set_ylim(0, 400)  # Adjust to your expected max count

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()

    ax1.legend(
        lines1,
        labels1,
        loc='upper left',
        borderaxespad=0.,
        fontsize='small',
        framealpha=1,
        edgecolor='black',
        fancybox=False,
    )

    ax1.set_title(r'\textbf{Comparing RF configurations in BICEPS}', y = 1.05)
    ax1.grid(True)

    plt.savefig(
        os.path.join(data_dir, 'Comparing RF configurations in BICEPS.png'),
        format='PNG',
        dpi=200,
        bbox_inches='tight'
    )
    plt.show()

def plot_radial_energies_from_files(data_dir, file_names, labels):
    """Plot two stacked zoom-in histograms for radial energies with separate binning for each zoom plot."""
    # Load all data and calculate means and SEM
    means = []
    sem_means = []

    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        _, energy_r = load_data(file_path)  # Load radial energy
        means.append(np.mean(energy_r))
        sem_means.append(np.std(energy_r, ddof=1) / np.sqrt(len(energy_r)))

    # Create a figure with two stacked zoom plots
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])  # Two equal-height zoom plots
    ax_zoom1 = fig.add_subplot(gs[0])  # Zoom plot for 1st and 3rd datasets
    ax_zoom2 = fig.add_subplot(gs[1])  # Zoom plot for 2nd and 4th datasets

    # Define colors and linestyles
    colors = ['tab:red', 'tab:blue', 'tab:purple', 'tab:orange']
    linestyles = ['-', '-', '-', '-']

    # Load data for 1st and 3rd datasets and calculate bins
    energy_r_zoom1 = []
    for i in [0, 2]:  # Indices for 1st and 3rd datasets
        file_path = os.path.join(data_dir, file_names[i])
        _, energy_r = load_data(file_path)
        energy_r_zoom1.extend(energy_r)
    bins_zoom1 = calculate_bins(energy_r_zoom1)  # Bins for 1st and 3rd datasets

    # Plot 1st and 3rd datasets in the first zoom plot
    for i in [0, 2]:
        file_path = os.path.join(data_dir, file_names[i])
        _, energy_r = load_data(file_path)
        label_with_data = (
            f"{labels[i]}\n"
            rf"$\langle E \rangle$: {means[i]:.2f} $\pm$ {sem_means[i]:.2f} meV"
        )
        ax_zoom1.hist(
            energy_r,
            bins=bins_zoom1,
            histtype='step',
            alpha=0.8,
            linewidth=2,
            color=colors[i],
            linestyle=linestyles[i],
            density=False,
            label=label_with_data
        )
    ax_zoom1.set_xlabel(r'$E$ (meV)')
    ax_zoom1.set_ylabel(r'Counts')
    ax_zoom1.set_xlim(0, 0.35)  # Adjustable x-axis limit for zoom
    ax_zoom1.set_ylim(0, 750)  # Adjust as needed
    ax_zoom1.set_title(r'\textbf{Radial Energies in IG}')
    ax_zoom1.legend(
        loc='upper right',
        borderaxespad=0.,
        fontsize='small',
        framealpha=1,
        edgecolor='black',
        fancybox=False,
    )
    ax_zoom1.grid(True)

    # Load data for 2nd and 4th datasets and calculate bins
    energy_r_zoom2 = []
    for i in [1, 3]:  # Indices for 2nd and 4th datasets
        file_path = os.path.join(data_dir, file_names[i])
        _, energy_r = load_data(file_path)
        energy_r_zoom2.extend(energy_r)
    bins_zoom2 = calculate_bins(energy_r_zoom2)  # Bins for 2nd and 4th datasets

    # Plot 2nd and 4th datasets in the second zoom plot
    for i in [1, 3]:
        file_path = os.path.join(data_dir, file_names[i])
        _, energy_r = load_data(file_path)
        label_with_data = (
            f"{labels[i]}\n"
            rf"$\langle E \rangle$: {means[i]:.2f} $\pm$ {sem_means[i]:.2f} meV"
        )
        ax_zoom2.hist(
            energy_r,
            bins=bins_zoom2,
            histtype='step',
            alpha=0.8,
            linewidth=2,
            color=colors[i],
            linestyle=linestyles[i],
            density=False,
            label=label_with_data
        )
    ax_zoom2.set_xlabel(r'$E$ (meV)')
    ax_zoom2.set_ylabel(r'Counts')
    ax_zoom2.set_xlim(0, 80)  # Adjustable x-axis limit for zoom
    ax_zoom2.set_ylim(0, 1000)  # Adjust as needed
    ax_zoom2.set_title(r'\textbf{Radial Energies at center BICEPS}')
    ax_zoom2.legend(
        loc='upper right',
        borderaxespad=0.,
        fontsize='small',
        framealpha=1,
        edgecolor='black',
        fancybox=False,
    )
    ax_zoom2.grid(True)

    plt.tight_layout()  # Adjust spacing between subplots
    plt.savefig(
        os.path.join(data_dir, 'Radial Energies Zoomed (Separate Binning).png'),
        format='PNG',
        dpi=200,
        bbox_inches='tight'
    )
    plt.show()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "STRIPE-IG-BICEPS_data")
    labels = [
        r"RF on one pair",
        r"RF on one pair",
        r"RF on both pairs",
        r"RF on both pairs"
    ]

    # List of files to process (adjust as needed)
    file_names = ['15042026/data 1/rawdata_IG.csv', '15042026/data 1/rawdata_BICEPS.csv', '15042026/data 2/rawdata_IG.csv', '15042026/data 2/rawdata_BICEPS.csv']

    print("Select which plots to generate:")
    print("1. Overlayed Axial Energies")
    print("2. Overlayed Radial Energies")
    print("0. Exit")

    while True:
        choice = input("Enter your choice (e.g., 1, 2, 0): ").strip()
        if not choice:
            break
        choices = choice.split(',')
        for c in choices:
            c = int(c.strip())
            if c == 1:
                plot_axial_energies_from_files(data_dir, file_names, labels)
            elif c == 2:
                plot_radial_energies_from_files(data_dir, file_names, labels)
            elif c == 0:
                return

if __name__ == "__main__":
    main()