import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math

from matplotlib.pyplot import close
from scipy.linalg import sqrtm
from scipy.optimize import curve_fit

plt.rcParams.update({
    'text.usetex': True,  # Enable LaTeX rendering
    'font.size': 24,      # Set global font size
    'font.family': 'serif'
})

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev)**2 / 2)

def calculate_fwhm(pos_r):
    # Fit a Gaussian to the histogram of radial positions
    hist, bin_edges = np.histogram(pos_r, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initial guess for amplitude, mean, and stddev
    amplitude_guess = np.max(hist)
    mean_guess = np.mean(pos_r)
    stddev_guess = np.std(pos_r)

    # Fit the Gaussian
    popt, _ = curve_fit(gaussian, bin_centers, hist, p0=[amplitude_guess, mean_guess, stddev_guess])
    _, mean, stddev = popt

    # Calculate FWHM
    fwhm = 2 * np.sqrt(2 * np.log(2)) * stddev
    return fwhm

def calculate_rms(pos_r):
    """
    Calculate the Root Mean Square (RMS) of a list of values.

    Args:
        pos_r (list or array-like): Input values.

    Returns:
        tuple: (RMS value, error on RMS)
    """
    r_squared = np.array([x ** 2 for x in pos_r])
    mean_r_squared = np.mean(r_squared)
    rms = math.sqrt(mean_r_squared)

    # Calculate error on RMS
    std_r_squared = np.std(r_squared, ddof=1)  # Sample standard deviation
    N = len(pos_r)
    error_rms = std_r_squared / (np.sqrt(N) * 2 * rms)

    return rms, error_rms

def calculate_percentile_radius(pos_r, percentile=95):
    """
    Calculate the radius under which a specified percentage of points lie.

    Args:
        pos_r (array-like): Array of radial distances.
        percentile (float): Percentile to calculate (e.g., 95 for 95% radius).

    Returns:
        float: Radius under which `percentile`% of points lie.
    """
    sorted_r = np.sort(pos_r)
    index = (percentile / 100) * (len(sorted_r) - 1)
    lower_index = int(np.floor(index))
    upper_index = int(np.ceil(index))

    # Linear interpolation if index is not an integer
    if lower_index == upper_index:
        return sorted_r[lower_index]
    else:
        lower_value = sorted_r[lower_index]
        upper_value = sorted_r[upper_index]
        fraction = index - lower_index
        return lower_value + fraction * (upper_value - lower_value)

def calculate_percentile_radius_with_error(pos_r, percentile=95, n_bootstraps=1000):
    """
    Calculate the 95% radius and its error using bootstrap resampling.

    Args:
        pos_r (array-like): Array of radial distances.
        percentile (float): Percentile to calculate (e.g., 95 for 95% radius).
        n_bootstraps (int): Number of bootstrap iterations.

    Returns:
        tuple: (95% radius, error on 95% radius)
    """
    # Calculate the original 95% radius
    radius_95 = calculate_percentile_radius(pos_r, percentile)

    # Bootstrap resampling
    bootstrap_radii = []
    n = len(pos_r)
    for _ in range(n_bootstraps):
        # Resample with replacement
        resampled_r = np.random.choice(pos_r, size=n, replace=True)
        # Calculate 95% radius for the resample
        bootstrap_radius = calculate_percentile_radius(resampled_r, percentile)
        bootstrap_radii.append(bootstrap_radius)

    # Error is the standard deviation of the bootstrap distribution
    error_95 = np.std(bootstrap_radii)

    return radius_95, error_95

# Load the data
def load_data(data_dir, file_name):
    raw_data = os.path.join(data_dir, file_name)
    positions_data = pd.read_csv(raw_data, skiprows=1, delimiter=',', header=None,
                                  names=["Pos_x", "Pos_y", "Pos_z", "Vel_x", "Vel_y", "Vel_z",
                                         "Energy_x", "Energy_y", "Energy_z", "ToF", "ion_number"])
    print(positions_data.head())

    # Group by ion_number and extract first and second occurrences
    grouped_pos_x = positions_data.groupby("ion_number")["Pos_x"]
    grouped_pos_y = positions_data.groupby("ion_number")["Pos_y"]

    pos_x_1 = (grouped_pos_x.nth(0).dropna().values - 16) * 10 ** 3  # First occurrence (um)
    pos_x_2 = (grouped_pos_x.nth(1).dropna().values - 16) * 10 ** 3  # Second occurrence (um)
    pos_x_3 = (grouped_pos_x.nth(2).dropna().values - 16) * 10 ** 3  # Second occurrence (um)
    pos_x_4 = (grouped_pos_x.nth(3).dropna().values - 16) * 10 ** 3  # Second occurrence (um)

    pos_y_1 = (grouped_pos_y.nth(0).dropna().values - 16) * 10 ** 3  # First occurrence (um)
    pos_y_2 = (grouped_pos_y.nth(1).dropna().values - 16) * 10 ** 3  # Second occurrence (um)
    pos_y_3 = (grouped_pos_y.nth(2).dropna().values - 16) * 10 ** 3  # Second occurrence (um)
    pos_y_4 = (grouped_pos_y.nth(3).dropna().values - 16) * 10 ** 3  # Second occurrence (um)

    bins = int(np.sqrt(len(pos_x_1)))

    return pos_x_1, pos_x_2, pos_x_3, pos_x_4, pos_y_1, pos_y_2, pos_y_3, pos_y_4, bins

def plot_beam_spots(data_dir, file_names):
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Add a large title at the top of the figure
    fig.suptitle(r'\textbf{Beam spot cross-sections while trapping in BICEPS}', y=0.95)

    # Adjust the spacing between subplots and margins
    plt.subplots_adjust(
        left=0.1,  # Left margin
        right=0.9,  # Right margin
        bottom=0.1,  # Bottom margin
        top=0.9,  # Top margin (leave space for the title)
        wspace=0.1,  # Horizontal space between subplots
        hspace=0.2  # Vertical space between subplots
    )

    # Define subplot labels
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']

    custom_notes = [
        r"First passing",
        r"Second passing",
        r"Third passing",
        r"Fourth passing"
    ]

    # Loop through the file names and plot each scatter plot
    for i, (file_name, label) in enumerate(zip(file_names, subplot_labels)):
        pos_x_1, pos_x_2, pos_x_3, pos_x_4, pos_y_1, pos_y_2, pos_y_3, pos_y_4, bins = load_data(data_dir, file_name)

        # Calculate RMS of radial positions for both datasets
        pos_r_1 = np.sqrt(pos_x_1 ** 2 + pos_y_1 ** 2)
        pos_r_2 = np.sqrt(pos_x_2 ** 2 + pos_y_2 ** 2)
        pos_r_3 = np.sqrt(pos_x_3 ** 2 + pos_y_3 ** 2)
        pos_r_4 = np.sqrt(pos_x_4 ** 2 + pos_y_4 ** 2)

        rms_pos_r_1, rms_error_pos_r_1 = calculate_rms(pos_r_1)
        rms_pos_r_2, rms_error_pos_r_2 = calculate_rms(pos_r_2)
        rms_pos_r_3, rms_error_pos_r_3 = calculate_rms(pos_r_3)
        rms_pos_r_4, rms_error_pos_r_4 = calculate_rms(pos_r_4)

        # Calculate 95% radius and error for each passing
        radius_95_1, error_95_1 = calculate_percentile_radius_with_error(pos_r_1, percentile=95)
        radius_95_2, error_95_2 = calculate_percentile_radius_with_error(pos_r_2, percentile=95)
        radius_95_3, error_95_3 = calculate_percentile_radius_with_error(pos_r_3, percentile=95)
        radius_95_4, error_95_4 = calculate_percentile_radius_with_error(pos_r_4, percentile=95)

        radius_95_list = [radius_95_1, radius_95_2, radius_95_3, radius_95_4]
        error_95_list = [error_95_1, error_95_2, error_95_3, error_95_4]

        print(radius_95_list)
        print(error_95_list)

        rms_pos_r_list = [rms_pos_r_1, rms_pos_r_2, rms_pos_r_3, rms_pos_r_4]
        rms_error_pos_r_list = [rms_error_pos_r_1, rms_error_pos_r_2, rms_error_pos_r_3, rms_error_pos_r_4]

        # Scatter plot for pos_x_1, pos_y_1 (First passing)
        axes[0].scatter(
            pos_x_1, pos_y_1,
            color='black',
            label='data points',
            alpha=0.5,
            s=5  # Marker size
        )

        # Scatter plot for pos_x_2, pos_y_2 (Second passing)
        axes[1].scatter(
            pos_x_2, pos_y_2,
            color='black',
            label='data points',
            alpha=0.5,
            s=5  # Marker size
        )

        # Scatter plot for pos_x_3, pos_y_3 (Third passing)
        axes[2].scatter(
            pos_x_3, pos_y_3,
            color='black',
            label='data points',
            alpha=0.5,
            s=5  # Marker size
        )

        # Scatter plot for pos_x_4, pos_y_4 (Fourth passing)
        axes[3].scatter(
            pos_x_4, pos_y_4,
            color='black',
            label='data points',
            alpha=0.5,
            s=5  # Marker size
        )

        # Draw circles for RMS and error shading
        theta = np.linspace(0, 2 * np.pi, 100)

        # Circle for RMS of first passing (blue)
        #x_circle_1 = rms_pos_r_1 * np.cos(theta)
        #y_circle_1 = rms_pos_r_1 * np.sin(theta)
        #axes[0].plot(x_circle_1, y_circle_1, color='red', linestyle='--', linewidth=1.5)

        # Shaded area for error of first passing (blue)
        x_circle_1_outer = (rms_pos_r_1 + rms_error_pos_r_1) * np.cos(theta)
        y_circle_1_outer = (rms_pos_r_1 + rms_error_pos_r_1) * np.sin(theta)
        x_circle_1_inner = (rms_pos_r_1 - rms_error_pos_r_1) * np.cos(theta)
        y_circle_1_inner = (rms_pos_r_1 - rms_error_pos_r_1) * np.sin(theta)
        axes[0].fill_between(np.concatenate([x_circle_1_outer, x_circle_1_inner[::-1]]),
                        np.concatenate([y_circle_1_outer, y_circle_1_inner[::-1]]),
                        color='red', alpha=0.5, label=r'RMS$_1$')

        # Circle for RMS of second passing (red)
        #x_circle_2 = rms_pos_r_2 * np.cos(theta)
        #y_circle_2 = rms_pos_r_2 * np.sin(theta)
        #axes[1].plot(x_circle_2, y_circle_2, color='red', linestyle='--', linewidth=1.5)

        # Shaded area for error of second passing (red)
        x_circle_2_outer = (rms_pos_r_2 + rms_error_pos_r_2) * np.cos(theta)
        y_circle_2_outer = (rms_pos_r_2 + rms_error_pos_r_2) * np.sin(theta)
        x_circle_2_inner = (rms_pos_r_2 - rms_error_pos_r_2) * np.cos(theta)
        y_circle_2_inner = (rms_pos_r_2 - rms_error_pos_r_2) * np.sin(theta)
        axes[1].fill_between(np.concatenate([x_circle_2_outer, x_circle_2_inner[::-1]]),
                        np.concatenate([y_circle_2_outer, y_circle_2_inner[::-1]]),
                        color='red', alpha=0.5, label=r'RMS$_2$')

        # Circle for RMS of third passing
        #x_circle_3 = rms_pos_r_3 * np.cos(theta)
        #y_circle_3 = rms_pos_r_3 * np.sin(theta)
        #axes[2].plot(x_circle_3, y_circle_3, color='red', linestyle='--', linewidth=1.5)

        # Shaded area for error of third passing
        x_circle_3_outer = (rms_pos_r_3 + rms_error_pos_r_3) * np.cos(theta)
        y_circle_3_outer = (rms_pos_r_3 + rms_error_pos_r_3) * np.sin(theta)
        x_circle_3_inner = (rms_pos_r_3 - rms_error_pos_r_3) * np.cos(theta)
        y_circle_3_inner = (rms_pos_r_3 - rms_error_pos_r_3) * np.sin(theta)
        axes[2].fill_between(np.concatenate([x_circle_3_outer, x_circle_3_inner[::-1]]),
                             np.concatenate([y_circle_3_outer, y_circle_3_inner[::-1]]),
                             color='red', alpha=0.5, label=r'RMS$_3$')

        # Circle for RMS of fourth passing
        #x_circle_4 = rms_pos_r_4 * np.cos(theta)
        #y_circle_4 = rms_pos_r_4 * np.sin(theta)
        #axes[3].plot(x_circle_4, y_circle_4, color='red', linestyle='--', linewidth=1.5)

        # Shaded area for error of fourth passing
        x_circle_4_outer = (rms_pos_r_4 + rms_error_pos_r_4) * np.cos(theta)
        y_circle_4_outer = (rms_pos_r_4 + rms_error_pos_r_4) * np.sin(theta)
        x_circle_4_inner = (rms_pos_r_4 - rms_error_pos_r_4) * np.cos(theta)
        y_circle_4_inner = (rms_pos_r_4 - rms_error_pos_r_4) * np.sin(theta)
        axes[3].fill_between(np.concatenate([x_circle_4_outer, x_circle_4_inner[::-1]]),
                             np.concatenate([y_circle_4_outer, y_circle_4_inner[::-1]]),
                             color='red', alpha=0.5, label=r'RMS$_4$')

        # Shaded area for 95% radius error (green)
        x_95_outer = (radius_95_list[0] + error_95_list[0]) * np.cos(theta)
        y_95_outer = (radius_95_list[1] + error_95_list[1]) * np.sin(theta)
        x_95_inner = (radius_95_list[2] - error_95_list[2]) * np.cos(theta)
        y_95_inner = (radius_95_list[3] - error_95_list[3]) * np.sin(theta)

        for i in range(0, 4):
            ax = axes[i]

            ax.fill_between(
                np.concatenate([x_95_outer, x_95_inner[::-1]]),
                np.concatenate([y_95_outer, y_95_inner[::-1]]),
                color='green', alpha=0.5, label = rf'$\mathrm{{95\% Radius_{i + 1}}}$'
            )

            ax.set_xlabel(r'$x$ ($\mu$m)')
            ax.set_ylabel(r'$y$ ($\mu$m)')
            ax.grid(True)
            ax.set_xlim([-60, 60])
            ax.set_ylim([-60, 60])
            ax.set_aspect('equal')  # Ensure circles are not distorted
            ax.legend(
                ncol = 1,
                loc='upper right',
                borderaxespad=0.,
                fontsize='small',
                framealpha=1,
                edgecolor='black',
                fancybox=False,
            )

            # Add subplot label (a), (b), etc.
            ax.text(
                0.015, 0.987,
                rf'\textbf{{{subplot_labels[i]}}}',  # Bold subplot label
                transform=ax.transAxes,
                fontsize=20,
                color='black',
                verticalalignment='top'
            )

            # Add custom note with RMS for both datasets
            note = (
                    rf'{custom_notes[i]}' + "\n" +
                    rf'$\mathrm{{RMS_{i + 1}}} = {rms_pos_r_list[i]:.2f} \pm {rms_error_pos_r_list[i]:.2f} \,\mathrm{{\mu m}}$' + "\n" +
                    rf'$\mathrm{{95\% Radius_{i + 1}}} = {radius_95_list[i]:.2f} \pm {error_95_list[i]:.2f} \,\mathrm{{\mu m}}$'
            )
            ax.text(
                0.013, 0.245,
                note,
                transform=ax.transAxes,
                fontsize=24,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=1)
            )

    # Save the figure
    fig.savefig(os.path.join(data_dir, 'beam_spots_scatter_BICEPS_trapping.png'), format='PNG', dpi=200, bbox_inches='tight')
    plt.show()

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "STRIPE-IG-BICEPS_data", "13052026")

file_names = ["data 1/rawdata_IG.csv"]

plot_beam_spots(data_dir, file_names)