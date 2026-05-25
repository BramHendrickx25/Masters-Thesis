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
    'font.size': 18,      # Set global font size
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
        float: RMS value.
    """
    squared = [x ** 2 for x in pos_r]
    mean_squared = sum(squared) / len(pos_r)
    rms = math.sqrt(mean_squared)
    return rms

# Load the data
def load_data(data_dir, file_name):
    raw_data = os.path.join(data_dir, file_name)
    positions_data = pd.read_csv(raw_data, skiprows=1, delimiter=',', header=None,
                                  names=["Pos_x", "Pos_y", "Pos_z", "Vel_x", "Vel_y", "Vel_z",
                                         "Energy_x", "Energy_y", "Energy_z", "ToF", "ion_number"])
    print(positions_data.head())

    pos_x = positions_data["Pos_x"] - 16
    pos_y = positions_data["Pos_y"] - 16
    pos_z = positions_data["Pos_z"]
    vel_x = positions_data["Vel_x"]
    vel_y = positions_data["Vel_y"]
    vel_z = positions_data["Vel_z"]
    energy_x = positions_data["Energy_x"]
    energy_y = positions_data["Energy_y"]
    energy_z = positions_data["Energy_z"] * 10**3
    tof = positions_data["ToF"]

    pos_r = np.sqrt(pos_x**2 + pos_y**2)
    vel_r = np.sqrt(vel_x**2 + vel_y**2)
    energy_r = np.sqrt(energy_x**2 + energy_y**2) * 10**3

    bins = int(np.sqrt(len(tof)))

    return pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, energy_x, energy_y, energy_z, tof, pos_r, vel_r, energy_r, bins

def plot_beam_spots(data_dir, file_names, custom_notes):
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Add a large title at the top of the figure
    fig.suptitle(r'\textbf{Beam Spot Cross-Sections for BICEPS RF configuration}', fontsize=24, y=0.98)

    # Adjust the spacing between subplots and margins
    plt.subplots_adjust(
        left=0.1,    # Left margin
        right=0.9,   # Right margin
        bottom=0.1,  # Bottom margin
        top=0.9,     # Top margin (leave space for the title)
        wspace=0.3,  # Horizontal space between subplots
        hspace=0.3   # Vertical space between subplots
    )

    # Define subplot labels
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']

    # Loop through the file names and plot each histogram
    for i, (file_name, custom_note, label) in enumerate(zip(file_names, custom_notes, subplot_labels)):
        pos_x, pos_y, _, _, _, _, _, _, _, _, _, _, _, bins = load_data(data_dir, file_name)

        # Calculate FWHM of radial positions
        pos_r = np.sqrt(pos_x**2 + pos_y**2)
        rms_pos_r = calculate_rms(pos_r)

        ax = axes[i]
        h = ax.hist2d(
            pos_x, pos_y,
            bins=bins,
            range=[[-0.025, 0.025], [-0.025, 0.025]],
            cmap='viridis',
            vmin=0
        )
        ax.set_xlabel(r'$x$ (mm)')
        ax.set_ylabel(r'$y$ (mm)')
        ax.grid(True)

        # Add colorbar to each subplot
        fig.colorbar(h[3], ax=ax, label=r'Counts ($N$)')

        # Add subplot label (a), (b), etc.
        ax.text(
            0.015, 0.987,
            rf'\textbf{{{label}}}',  # Bold subplot label
            transform=ax.transAxes,
            fontsize=20,
            color = 'white',
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.0)  # Transparent background
        )

        # Add custom note with FWHM appended
        note = rf'\shortstack{{{custom_note} \\ $\mathrm{{RMS}} = {rms_pos_r:.3f}\,\mathrm{{mm}}$}}'
        ax.text(
            0.015, 0.134,
            note,
            transform=ax.transAxes,
            fontsize=16,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8)
        )

    # Save the figure
    fig.savefig(os.path.join(data_dir, 'beam_spots_BICEPS_RF_config.png'), format='PNG', dpi=200, bbox_inches='tight')
    plt.show()

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "STRIPE-IG-BICEPS_data", "06052026")

file_names = ["data 1/rawdata_IG.csv", "data 2/rawdata_IG.csv", "data 3/rawdata_IG.csv", "data 4/rawdata_IG.csv"]
custom_notes = [
    r"single pair RF (in IG)",
    r"single pair RF (in IG)",
    r"double pair RF (in IG)",
    r"double pair RF (in IG)"
]
plot_beam_spots(data_dir, file_names, custom_notes)
