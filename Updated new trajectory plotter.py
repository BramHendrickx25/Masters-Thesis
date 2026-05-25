import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'text.usetex': True,  # Enable LaTeX rendering
    'font.size': 18,      # Set global font size
    'font.family': 'serif'
})

def load_and_process_data(data_path):
    """
    Loads and processes the data from the given file path.
    New format: x, y, z, x_velocity, y_velocity, z_velocity, ToF.

    Parameters:
        data_path (str): Path to the data file.

    Returns:
        trajectories (numpy.ndarray): Processed trajectories array (positions only).
        z_min (float): Minimum z value.
        z_max (float): Maximum z value.
        x_min_adj (float): Adjusted minimum x value.
        x_max_adj (float): Adjusted maximum x value.
        y_min_adj (float): Adjusted minimum y value.
        y_max_adj (float): Adjusted maximum y value.
    """
    data = np.loadtxt(data_path, delimiter=',', dtype=float, skiprows=1)

    n_ions = 10
    n_features = 7  # x, y, z, x_velocity, y_velocity, z_velocity, ToF
    expected_rows_per_timestep = n_ions

    total_rows = data.shape[0]
    if total_rows % expected_rows_per_timestep != 0:
        n_complete_timesteps = total_rows // expected_rows_per_timestep
        data = data[:n_complete_timesteps * expected_rows_per_timestep]
        print(f"Trimmed data to {n_complete_timesteps} complete timesteps.")

    n_timesteps = len(data) // n_ions
    data_reshaped = data.reshape(n_timesteps, n_ions, n_features)

    # Extract only the position data (first 3 columns) for trajectories
    trajectories = data_reshaped[:, :, :3].transpose(1, 0, 2)

    # Sort trajectories by initial z position (descending)
    initial_z = trajectories[:, 0, 2]
    sorted_indices = np.argsort(-initial_z)
    trajectories = trajectories[sorted_indices]

    # Calculate bounds for visualization
    z_min = np.min(trajectories[:, :, 2]) - 0.01
    z_max = np.max(trajectories[:, :, 2]) + 0.01
    z_range = z_max - z_min

    x_min, x_max = np.min(trajectories[:, :, 0]), np.max(trajectories[:, :, 0])
    y_min, y_max = np.min(trajectories[:, :, 1]), np.max(trajectories[:, :, 1])

    x_mean = (x_min + x_max) / 2
    y_mean = (y_min + y_max) / 2
    x_min_adj = x_mean - z_range / 2
    x_max_adj = x_mean + z_range / 2
    y_min_adj = y_mean - z_range / 2
    y_max_adj = y_mean + z_range / 2

    velocities = data_reshaped[:, :, 3:6].transpose(1, 0, 2)  # x, y, z velocities
    tof = data_reshaped[:, :, 6].transpose(1, 0)  # Time of Flight
    return trajectories, velocities, tof, z_min, z_max, x_min_adj, x_max_adj, y_min_adj, y_max_adj

def plot_trajectory_plots(trajectories, z_min, z_max, x_min_adj, x_max_adj, y_min_adj, y_max_adj, save_dir):
    """
    Plots the 3D and 2D trajectory plots with a single legend outside the plots.
    """
    colors = plt.colormaps['tab10'].resampled(10)

    fig = plt.figure(figsize=(16, 14))

    # Create a list to store legend handles and labels
    handles = []
    labels = []

    # 3D Plot
    ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
    for i in range(10):
        line, = ax_3d.plot(
            trajectories[i, :, 2],
            trajectories[i, :, 1],
            trajectories[i, :, 0],
            color=colors(i),
            linewidth=1
        )
        handles.append(line)
        labels.append(f'Ion {i+1}')
    ax_3d.set_xlabel('Z (mm)')
    ax_3d.set_ylabel('Y (mm)')
    ax_3d.set_zlabel('X (mm)')
    ax_3d.set_title('3D Trajectories')
    ax_3d.set_xlim([z_min, z_max])
    ax_3d.set_ylim([y_min_adj, y_max_adj])
    ax_3d.set_zlim([x_min_adj, x_max_adj])

    # X-Z Plot
    ax_xz = fig.add_subplot(2, 2, 2)
    for i in range(10):
        ax_xz.plot(
            trajectories[i, :, 2],
            trajectories[i, :, 0],
            color=colors(i),
            linewidth=1
        )
    ax_xz.set_xlabel('Z (mm)')
    ax_xz.set_ylabel('X (mm)')
    ax_xz.set_title('X-Z Trajectories')
    ax_xz.set_xlim([z_min, z_max])
    ax_xz.set_ylim([x_min_adj, x_max_adj])
    ax_xz.set_aspect('equal')

    # Y-Z Plot
    ax_yz = fig.add_subplot(2, 2, 3)
    for i in range(10):
        ax_yz.plot(
            trajectories[i, :, 2],
            trajectories[i, :, 1],
            color=colors(i),
            linewidth=1
        )
    ax_yz.set_xlabel('Z (mm)')
    ax_yz.set_ylabel('Y (mm)')
    ax_yz.set_title('Y-Z Trajectories')
    ax_yz.set_xlim([z_min, z_max])
    ax_yz.set_ylim([y_min_adj, y_max_adj])
    ax_yz.set_aspect('equal')

    # X-Y Plot
    ax_xy = fig.add_subplot(2, 2, 4)
    for i in range(10):
        ax_xy.plot(
            trajectories[i, :, 0],
            trajectories[i, :, 1],
            color=colors(i),
            linewidth=1
        )
    ax_xy.set_xlabel('X (mm)')
    ax_xy.set_ylabel('Y (mm)')
    ax_xy.set_title('X-Y Trajectories')
    ax_xy.set_xlim([x_min_adj, x_max_adj])
    ax_xy.set_ylim([y_min_adj, y_max_adj])
    ax_xy.set_aspect('equal')

    # Add a single legend to the figure, positioned outside the plots
    fig.legend(
        handles=handles,
        labels=labels,
        loc='center left',
        bbox_to_anchor=(0.92, 0.5),  # Position the legend outside the plots
        borderaxespad=0.,
        fontsize='small',
        framealpha=1,
        edgecolor='black',
        fancybox=False,
    )

    # Adjust the layout to make room for the legend
    plt.subplots_adjust(right=0.95)  # Leave space on the right for the legend

    plt.savefig(os.path.join(save_dir, 'trajectories.png'), format='PNG', dpi=200, bbox_inches='tight')
    plt.close()

def plot_xz_heatmap(trajectories, save_dir, bins=int(np.sqrt(480000))):
    """
    Plots a heatmap of ion positions in the x-z plane and a histogram of bin counts.
    """
    x_positions = trajectories[:, :, 0].flatten()
    z_positions = trajectories[:, :, 2].flatten()

    x_mean = np.mean(x_positions)
    z_range = np.max(z_positions) - np.min(z_positions)
    x_min = x_mean - z_range/2
    x_max = x_mean + z_range/2
    heatmap, zedges, xedges = np.histogram2d(
        z_positions, x_positions, bins=bins, range=[[z_positions.min() - 0.02, z_positions.max() + 0.02], [x_min, x_max]]
    )

    # Plot the heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    vmin = 2000
    vmax = 60000
    im = ax1.imshow(
        heatmap.T,
        extent=[zedges[0], zedges[-1], xedges[0], xedges[-1]],
        origin='lower',
        aspect='auto',
        cmap='viridis',
        vmin=vmin,
        vmax=vmax
    )

    ax1.set_xlabel('Z (mm)')
    ax1.set_ylabel('X (mm)')
    ax1.set_aspect('equal')
    ax1.set_title('Heatmap of Ion Positions (X-Z Plane)')

    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label('Density')

    # Flatten the heatmap and filter out bins with counts <= 1
    bin_counts = heatmap.flatten()
    bin_counts_filtered = bin_counts[bin_counts > 0]  # Exclude counts <= 1

    # Plot the histogram of the filtered bin counts
    ax2.hist(bin_counts_filtered, bins=bins, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Density')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Density ($> 0$)')
    ax2.axvline(x=vmin, color='red', linestyle='--', label=f'Density = {vmin}')
    ax2.axvline(x=vmax, color='green', linestyle='--', label=f'Density = {vmax}')
    ax2.legend(loc='upper right', fancybox=False, framealpha=1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'xz_heatmap_with_histogram.png'), format='PNG', dpi=200, bbox_inches='tight')
    plt.close()

def plot_xy_heatmap(trajectories, save_dir, bins=int(np.sqrt(48000))):
    """
    Plots a heatmap of ion positions in the x-y plane.

    Parameters:
        trajectories (numpy.ndarray): Processed trajectories array.
        save_dir (str): Directory to save the heatmap plot.
        bins (int): Number of bins for the 2D histogram (default: 100).
    """
    x_positions = trajectories[:, :, 0].flatten()
    y_positions = trajectories[:, :, 1].flatten()

    heatmap, xedges, yedges = np.histogram2d(
        x_positions, y_positions, bins=bins, range=[[x_positions.min() - 0.001, x_positions.max() + 0.001], [y_positions.min() - 0.001, y_positions.max() + 0.001]]
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        heatmap.T,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin='lower',
        aspect='auto',
        cmap='viridis',
    )

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Heatmap of Ion Positions (X-Y Plane)')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Density')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'xy_heatmap.png'), format='PNG', dpi=200, bbox_inches='tight')
    plt.close()

def calculate_mean_positions_and_distances(trajectories):
    """
    Calculate the mean positions of each ion, their standard deviations, and the pairwise distances
    between these mean positions, including the propagated uncertainty in the distance matrix.

    Parameters:
        trajectories (numpy.ndarray): Processed trajectories array of shape (10, n_timesteps, 3).

    Returns:
        mean_positions (numpy.ndarray): Array of shape (10, 3) containing the mean (x, y, z) position for each ion.
        std_positions (numpy.ndarray): Array of shape (10, 3) containing the standard deviation of (x, y, z) for each ion.
        distance_matrix (numpy.ndarray): Array of shape (10, 10) containing the Euclidean distances between mean positions of all ion pairs.
        distance_errors (numpy.ndarray): Array of shape (10, 10) containing the propagated uncertainty in the distances.
    """
    # Calculate mean and standard deviation of positions for each ion
    mean_positions = np.mean(trajectories, axis=1)  # Shape: (10, 3)
    std_positions = np.std(trajectories, axis=1)    # Shape: (10, 3)

    # Calculate pairwise distances between mean positions
    n_ions = mean_positions.shape[0]
    distance_matrix = np.zeros((n_ions, n_ions))
    distance_errors = np.zeros((n_ions, n_ions))

    for i in range(n_ions):
        for j in range(n_ions):
            delta = mean_positions[i] - mean_positions[j]
            distance_matrix[i, j] = np.linalg.norm(delta)

            # Propagate uncertainty in the distance
            if distance_matrix[i, j] > 0:  # Avoid division by zero for i == j
                unit_vector = delta / distance_matrix[i, j]
                # Variance of the distance: sum over x, y, z of (unit_vector * std)^2
                variance = np.sum((unit_vector ** 2) * (std_positions[i] ** 2 + std_positions[j] ** 2))
                distance_errors[i, j] = np.sqrt(variance)
            else:
                distance_errors[i, j] = 0.0  # No error for self-distance

    return mean_positions, std_positions, distance_matrix, distance_errors

def plot_z_histogram(trajectories, save_dir, bins=100):
    """
    Plots a histogram of ion positions along the z-axis.

    Parameters:
        trajectories (numpy.ndarray): Processed trajectories array.
        save_dir (str): Directory to save the histogram plot.
        bins (int): Number of bins for the histogram (default: 100).
    """
    # Extract z positions for all ions and timesteps
    z_positions = trajectories[:, :, 2].flatten()

    # Plot histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(z_positions, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)

    ax.set_xlabel('Z Position (mm)')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Ion Positions Along Z-Axis')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'z_histogram.png'), format='PNG', dpi=200, bbox_inches='tight')
    plt.close()

def plot_first_ion_xz_heatmap(trajectories, save_dir, ion_index=0, bins=int(np.sqrt(4800))):
    """
    Plots a 2D heatmap of the first ion's positions in the x-z plane with a square aspect ratio.

    Parameters:
        trajectories (numpy.ndarray): Processed trajectories array of shape (10, n_timesteps, 3).
        save_dir (str): Directory to save the heatmap plot.
        ion_index (int): Index of the ion to plot (default: 0, the first ion).
        bins (int): Number of bins for the 2D histogram (default: sqrt(480000)).
    """
    # Extract x and z positions for the specified ion
    x_positions = trajectories[ion_index, :, 0]
    z_positions = trajectories[ion_index, :, 2]

    # Calculate the range for z and x
    z_min, z_max = np.min(z_positions)-0.001, np.max(z_positions)+0.001
    x_min, x_max = np.min(x_positions), np.max(x_positions)

    # Use the larger range for both axes to ensure square aspect ratio
    max_range = max(z_max - z_min, x_max - x_min)
    z_center = (z_min + z_max) / 2
    x_center = (x_min + x_max) / 2

    z_min_adj = z_center - max_range / 2
    z_max_adj = z_center + max_range / 2
    x_min_adj = x_center - max_range / 2
    x_max_adj = x_center + max_range / 2

    # Create 2D histogram
    heatmap, zedges, xedges = np.histogram2d(
        z_positions, x_positions,
        bins=bins,
        range=[[z_min_adj, z_max_adj], [x_min_adj, x_max_adj]]
    )

    # Plot the heatmap
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    vmin = 2000
    vmax = 100000

    im = ax.imshow(
        heatmap.T,
        extent=[zedges[0], zedges[-1], xedges[0], xedges[-1]],
        origin='lower',
        aspect='equal',  # Force square pixels
        cmap='viridis',
        vmin = vmin,
        vmax = vmax
    )

    ax.set_xlabel('Z (mm)')
    ax.set_ylabel('X (mm)')
    ax.set_title(f'Heatmap of Ion {ion_index + 1} Positions (X-Z Plane)')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Density')

    # Flatten the heatmap and filter out bins with counts <= 1
    bin_counts = heatmap.flatten()
    bin_counts_filtered = bin_counts[bin_counts > 0]  # Exclude counts <= 1

    # Plot the histogram of the filtered bin counts
    ax2.hist(bin_counts_filtered, bins=bins, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Density')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Density ($> 0$)')
    ax2.axvline(x=vmin, color='red', linestyle='--', label=f'Density = {vmin}')
    ax2.axvline(x=vmax, color='green', linestyle='--', label=f'Density = {vmax}')
    ax2.legend(loc='upper right', fancybox=False, framealpha=1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'xz_single_ion_heatmap_with_histogram.png'), format='PNG', dpi=200, bbox_inches='tight')
    plt.close()

def plot_ion_positions_vs_time(trajectories, tof, save_dir, axis='z'):
    """
    Plots the x, y, or z positions of all 10 ions as a function of ToF.
    Each ion is plotted with a distinct color.

    Parameters:
        trajectories (numpy.ndarray): Processed trajectories array of shape (10, n_timesteps, 3).
        tof (numpy.ndarray): Time of Flight array of shape (10, n_timesteps).
        save_dir (str): Directory to save the plot.
        axis (str): Which axis to plot ('x', 'y', or 'z'). Default: 'z'.
    """
    # Validate axis input
    axis = axis.lower()
    if axis not in ['x', 'y', 'z']:
        raise ValueError("axis must be 'x', 'y', or 'z'.")

    # Map axis to the correct column in trajectories
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_index = axis_map[axis]

    # Extract positions for all ions along the specified axis
    positions = trajectories[:, :, axis_index]  # Shape: (10, n_timesteps)

    # Create a color map for the 10 ions
    colors = plt.colormaps['tab10'].resampled(10)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(10):
        ax.plot(
            tof[i],
            positions[i],
            color=colors(i),
            label=f'Ion {i+1}',
            linewidth=1.5
        )

    ax.set_xlabel('Time of Flight (usec)')
    ax.set_ylabel(f'{axis.upper()} Position (mm)')
    ax.set_title(f'Position of Ions vs. Time of Flight ({axis.upper()}-Axis)')
    ax.legend(ncol=5, handletextpad=0.1, borderpad=0.1, fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'ion_positions_vs_tof_{axis}.png'), format='PNG', dpi=200, bbox_inches='tight')
    plt.close()

def plot_single_ion_positions_vs_time(trajectories, tof, save_dir, ion_index=0, axis='z'):
    """
    Plots the x, y, or z position of a single ion as a function of ToF.

    Parameters:
        trajectories (numpy.ndarray): Processed trajectories array of shape (10, n_timesteps, 3).
        tof (numpy.ndarray): Time of Flight array of shape (10, n_timesteps).
        save_dir (str): Directory to save the plot.
        ion_index (int): Index of the ion to plot (default: 0, the first ion).
        axis (str): Which axis to plot ('x', 'y', or 'z'). Default: 'z'.
    """
    # Validate inputs
    if ion_index < 0 or ion_index >= trajectories.shape[0]:
        raise ValueError(f"ion_index must be between 0 and {trajectories.shape[0] - 1}.")

    axis = axis.lower()
    if axis not in ['x', 'y', 'z']:
        raise ValueError("axis must be 'x', 'y', or 'z'.")

    # Map axis to the correct column in trajectories
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_index = axis_map[axis]

    # Extract positions and ToF for the specified ion
    positions = trajectories[ion_index, :, axis_index]
    time = tof[ion_index]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        time,
        positions,
        color='blue',
        linewidth=1.5,
        label=f'Ion {ion_index + 1} {axis.upper()} Position'
    )

    ax.set_xlabel('Time of Flight (usec)')
    ax.set_ylabel(f'{axis.upper()} Position (mm)')
    ax.set_title(f'Position of Ion {ion_index + 1} vs. Time of Flight ({axis.upper()}-Axis)')
    ax.legend(fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'ion_{ion_index + 1}_positions_vs_tof_{axis}.png'), format='PNG', dpi=200, bbox_inches='tight')
    plt.close()

def calculate_beam_longitudinal_emittance(trajectories, velocities, ion_mass_kg=None):
    """
    Calculate the longitudinal emittance for the entire beam using pooled
    (z, gamma*beta_z) phase-space variances.

    SCOPE: this is intended for the EXTRACTED beam (after ejection, where
    ions have a substantial centroid velocity along z and the transport-beam
    emittance formalism applies). Do NOT use this on the in-trap crystal:
    pooling all ions' z positions makes sigma^2(z) dominated by the chain
    length, not by thermal motion. For the in-trap analysis use
    per_ion_axial_emittance() instead.

    Parameters:
        trajectories (numpy.ndarray): Shape (N, n_timesteps, 3) - x, y, z positions in mm.
        velocities (numpy.ndarray): Shape (N, n_timesteps, 3) - x, y, z velocities in mm/s.
        ion_mass_kg (float, optional): Mass of the ion in kg. If provided, normalized emittance is calculated.

    Returns:
        emittance (float): Longitudinal emittance for the entire beam in m (or unitless if normalized).
    """
    # Constants
    c = 299792458  # Speed of light in m/s

    # Flatten z-positions and z-velocities across all ions and timesteps
    z_positions = trajectories[:, :, 2].flatten() * 10 ** -3  # Convert mm to m
    z_velocities = velocities[:, :, 2].flatten() * 10 ** 3   # Convert mm/usec to m/s

    # Calculate relativistic factors
    v_z_mean = np.sqrt(np.mean(z_velocities**2))  # Mean velocity in m/s
    beta = v_z_mean / c
    gamma = 1 / np.sqrt(1 - beta ** 2)

    # Normalized momentum (γβ_z)
    gamma_beta_z = gamma * (z_velocities / c)

    # Calculate means
    mean_z = np.mean(z_positions)
    mean_gamma_beta_z = np.mean(gamma_beta_z)

    # Calculate variances
    sigma_z_sq = np.var(z_positions, ddof=0)  # Variance of z (m²)
    sigma_gamma_beta_z_sq = np.var(gamma_beta_z, ddof=0)  # Variance of γβ_z (unitless)

    # Calculate covariance
    cov_z_gamma_beta_z = np.mean((z_positions - mean_z) * (gamma_beta_z - mean_gamma_beta_z))  # Covariance (m)

    # Calculate RMS emittance
    emittance_RMS = np.sqrt(sigma_z_sq * sigma_gamma_beta_z_sq - cov_z_gamma_beta_z ** 2)

    print(f"Mean z-velocity: {v_z_mean:.4f} m/s")
    print(f"Variance of z: {sigma_z_sq:.4f} m²")
    print(f"Variance of γβ_z: {sigma_gamma_beta_z_sq:.4f}")
    print(f"Covariance (z, γβ_z): {cov_z_gamma_beta_z:.4f} m")
    print(f"RMS Emittance: {emittance_RMS:.4f} m")

    return emittance_RMS

def calculate_beam_transversal_emittance(trajectories, velocities, ion_mass_kg=None):
    """
    Per-axis transverse RMS emittances epsilon_x and epsilon_y, normalised
    (i.e. using gamma*beta as the momentum coordinate).

    Returns (eps_x, eps_y), each in m (geometric m*rad, with rad dimensionless).

    Replaces the previous 4D sqrt(det(Sigma)) version, which actually returned
    eps_x * eps_y with units (m*rad)^2 = m^2 and was therefore not a 1D
    emittance compatible with the temperature formulas downstream.

    SCOPE: as with the longitudinal version, this is intended for the
    EXTRACTED beam. For in-trap radial temperatures use kinetic_temperature_per_axis()
    with stroboscopic sampling to remove RF micromotion.
    """
    # Constants
    c = 299792458  # Speed of light in m/s

    # Flatten positions and velocities across all ions and timesteps
    x_positions = trajectories[:, :, 0].flatten() * 10 ** -3  # mm to m
    y_positions = trajectories[:, :, 1].flatten() * 10 ** -3  # mm to m

    x_velocities = velocities[:, :, 0].flatten() * 10 ** 3   # mm/usec to m/s
    y_velocities = velocities[:, :, 1].flatten() * 10 ** 3   # mm/usec to m/s
    z_velocities = velocities[:, :, 2].flatten() * 10 ** 3   # mm/usec to m/s

    # Speed and Lorentz factor per particle (for any reasonable trapped-ion
    # case beta << 1, so gamma is essentially 1, but kept for consistency).
    v_magnitude = np.sqrt(x_velocities**2 + y_velocities**2 + z_velocities**2)
    beta = v_magnitude / c
    gamma = 1 / np.sqrt(1 - beta**2)

    # Normalized transverse momenta
    gamma_beta_x = gamma * (x_velocities / c)
    gamma_beta_y = gamma * (y_velocities / c)

    # Per-axis RMS emittance: eps = sqrt(<x^2><p^2> - <xp>^2)
    cov_x = np.cov(x_positions, gamma_beta_x)
    cov_y = np.cov(y_positions, gamma_beta_y)
    det_x = cov_x[0, 0] * cov_x[1, 1] - cov_x[0, 1] ** 2
    det_y = cov_y[0, 0] * cov_y[1, 1] - cov_y[0, 1] ** 2
    eps_x = np.sqrt(max(det_x, 0.0))
    eps_y = np.sqrt(max(det_y, 0.0))

    return eps_x, eps_y

def calculate_temperature_from_emittance(
    emittance,
    omega_z,
    ion_mass_kg,
    velocities
):
    """
    Convert a longitudinal emittance to a temperature using the
    BEAM-TRANSPORT convention, where sqrt(2 m <E_k>) relates velocity
    spread to energy spread via dE = m v dv and ln(20) is a 90 %
    containment factor.

    SCOPE: this is intended for the EXTRACTED beam (after ejection,
    where ions carry a substantial centroid kinetic energy <E_k>).
    Do NOT use this for the in-trap crystal: in the trap <v> = 0
    and the proper relation is the simple harmonic one,
    T = m * eps_geom * omega / k_B,
    available as temperature_from_in_trap_emittance().

    Parameters:
        emittance (float): Longitudinal emittance in kgm²/s.
        omega_z (float): Longitudinal oscillation frequency in rad/s.
        ion_mass_kg (float): Ion mass in kg.
        velocities (ndarray): Array of velocities in mm/s (shape: N x M x 3).

    Returns:
        temperature (float): Temperature in Kelvin.
    """
    k_B = 1.380649e-23  # Boltzmann constant in J/K

    # Extract velocities in m/s (flatten and convert from mm/usec to m/s)
    x_velocities = velocities[:, :, 0].flatten() * 10 ** 3  # Convert mm/usec to m/s
    y_velocities = velocities[:, :, 1].flatten() * 10 ** 3  # Convert mm/usec to m/s
    z_velocities = velocities[:, :, 2].flatten() * 10 ** 3  # Convert mm/usec to m/s

    # Calculate kinetic energy for each particle (J)
    kinetic_energies = 0.5 * ion_mass_kg * (x_velocities**2 + y_velocities**2 + z_velocities**2)

    # Mean kinetic energy
    mean_kinetic_energy = np.mean(kinetic_energies)
    print(mean_kinetic_energy)

    # Calculate temperature
    temperature = (emittance * omega_z) * np.sqrt(2 * ion_mass_kg * mean_kinetic_energy) / (2 * np.pi * np.log(20) * k_B)

    # Convert to float explicitly
    temperature = float(temperature)

    return temperature


# ----------------------------------------------------------------------------
# In-trap temperature diagnostics (added after code review).
#
# The functions below are appropriate for the CRYSTAL in the trap, where
# <v> = 0 and the system is a set of coupled harmonic oscillators in
# thermal (or near-thermal) equilibrium. They avoid two pitfalls of the
# original code:
#   (a) pooling all ions' positions, which makes sigma^2(z) reflect chain
#       length rather than thermal spread;
#   (b) using the transport-beam temperature formula, which assumes a
#       non-zero centroid velocity.
#
# For radial directions (x, y) the velocity contains coherent RF
# micromotion that is not thermal. To extract the secular (true thermal)
# component, pass `strobe_indices` to the kinetic-temperature routine:
# indices that sample at integer multiples of the RF period.
# ----------------------------------------------------------------------------

def kinetic_temperature_per_axis (velocities, ion_mass_kg, strobe_indices=None):
    """
    Direct per-axis temperature from velocity variance:
        T_i = m * <(v_i - <v_i>)^2> / k_B
    computed PER ION and then averaged across ions (so chain motion or
    drift of the whole crystal does not enter sigma^2).

    Also computes the transverse temperature (x-y plane) accounting for covariance.

    Parameters
    ----------
    velocities : ndarray, shape (N_ions, n_t, 3)
        SIMION velocities in mm/usec.
    ion_mass_kg : float
        Ion mass in kg.
    strobe_indices : array-like of ints, optional
        Time-step indices to subsample (e.g. sampling at integer multiples
        of the RF period to remove radial micromotion). If None, all
        timesteps are used.

    Returns
    -------
    (T_x, T_y, T_z, T_xy) : tuple of floats
        Temperatures along each axis (x, y, z) and the transverse temperature (x-y plane), in K.
    """
    k_B = 1.380649e-23
    v = velocities * 1.0e3  # mm/usec -> m/s
    if strobe_indices is not None:
        v = v[:, strobe_indices, :]

    Ts = np.zeros(3)
    for axis in range(3):
        # variance per ion (over time), then mean across ions
        per_ion_var = np.var(v[:, :, axis], axis=1)
        Ts[axis] = ion_mass_kg * float(np.mean(per_ion_var)) / k_B

    # Transverse temperature (x-y plane) with covariance
    v_xy = v[:, :, :2]  # Extract x and y velocities (shape: N_ions, n_t, 2)

    # Compute covariance matrix for each ion (over time)
    # Shape: (N_ions, 2, 2)
    cov_matrices = np.zeros((v_xy.shape[0], 2, 2))
    for i in range(v_xy.shape[0]):
        # Center the velocities (subtract mean over time for each ion)
        v_centered = v_xy[i] - np.mean(v_xy[i], axis=0)
        # Compute covariance matrix for this ion
        cov_matrices[i] = np.cov(v_centered, rowvar=False)

    # Average covariance matrix across all ions
    avg_cov = np.mean(cov_matrices, axis=0)

    # Transverse temperature: T_xy = m * sqrt(det(avg_cov)) / k_B
    # Note: This is a simplified approach. For true emittance, you'd need positions and momenta.
    # Here, we assume the covariance of velocities is a proxy for the correlation in the transverse plane.
    det_cov = np.linalg.det(avg_cov)
    T_xy = ion_mass_kg * np.sqrt(det_cov) / k_B

    return Ts[0], Ts[1], Ts[2], T_xy


def per_ion_axial_emittance(trajectories, velocities):
    """
    Axial geometric RMS emittance computed per ion, then averaged.

    Each ion's time-mean (z_i, v_{z,i}) is subtracted before computing
    variances, so the result reflects thermal-amplitude motion of the
    ion about its equilibrium position in the chain, not the chain
    geometry itself.

    Returns
    -------
    eps_mean : float
        Mean per-ion geometric RMS emittance, in m (= m*rad).
    eps_std : float
        Spread across ions, same units.
    """
    z_pos = trajectories[:, :, 2] * 1.0e-3   # mm -> m
    z_vel = velocities[:, :, 2] * 1.0e3      # mm/usec -> m/s

    N = z_pos.shape[0]
    eps = np.zeros(N)
    for i in range(N):
        z = z_pos[i] - np.mean(z_pos[i])
        v = z_vel[i] - np.mean(z_vel[i])
        sz2 = np.mean(z * z)
        sv2 = np.mean(v * v)
        sz_v = np.mean(z * v)
        eps[i] = np.sqrt(max(sz2 * sv2 - sz_v * sz_v, 0.0))
    return float(np.mean(eps)), float(np.std(eps))


def temperature_from_in_trap_emittance(emittance, omega, ion_mass_kg):
    """
    Convert a per-axis geometric RMS emittance to temperature for a
    harmonic trap in thermal equilibrium:
        T = m * eps * omega / k_B
    (Cross-check for kinetic_temperature_per_axis. The two definitions
    should agree to within statistical noise once the system is
    equilibrated.)
    """
    k_B = 1.380649e-23
    return float(ion_mass_kg * emittance * omega / k_B)


def stroboscopic_indices(tof, rf_period_us):
    """
    Return time-step indices that sample at integer multiples of the RF
    period. Use to suppress radial micromotion when computing radial T.

    Parameters
    ----------
    tof : ndarray, shape (N_ions, n_t)
        Time-of-flight per ion per step, in usec (SIMION default).
    rf_period_us : float
        RF period in usec (= 1 / f_RF).

    Returns
    -------
    idx : ndarray of int
        Indices of timesteps closest to integer multiples of rf_period_us
        from the start.
    """
    t = tof[0]  # use ion 0's clock; all ions share the same SIMION time
    t0 = t[0]
    n_periods = int(np.floor((t[-1] - t0) / rf_period_us))
    targets = t0 + np.arange(n_periods + 1) * rf_period_us
    idx = np.array([int(np.argmin(np.abs(t - tg))) for tg in targets])
    return np.unique(idx)


# Example usage:
# mean_positions, distance_matrix = calculate_mean_positions_and_distances(trajectories)
# print("Mean positions:\n", mean_positions)
# print("Distance matrix:\n", distance_matrix)

# Example usage
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join("C:/Users/hendr/OneDrive - KU Leuven/Master of Physics/STRIPE-IG-BICEPS/17052026/data 6")
data_path = os.path.join(data_dir, "trajectories.csv")

# Load and process the data
trajectories, velocities, tof, z_min, z_max, x_min_adj, x_max_adj, y_min_adj, y_max_adj = load_and_process_data(data_path)

# Plot trajectory plots
# plot_trajectory_plots(trajectories, z_min, z_max, x_min_adj, x_max_adj, y_min_adj, y_max_adj, data_dir)

# Plot heatmap
# plot_xz_heatmap(trajectories, data_dir)

mean_positions, std_positions, distance_matrix, distance_errors = calculate_mean_positions_and_distances(trajectories)
print("Mean Positions:\n", mean_positions)
print("Standard Deviations:\n", std_positions)
print("Distance Matrix:\n", distance_matrix)
print("Distance Errors:\n", distance_errors)

# Save to a file if needed
# np.savetxt(os.path.join(data_dir, 'mean_positions.csv'), mean_positions, delimiter=',', header='x,y,z', comments='')
# np.savetxt(os.path.join(data_dir, 'distance_matrix.csv'), distance_matrix, delimiter=',', comments='')

# PLot heatmap xy
# plot_xy_heatmap(trajectories, data_dir)

# Plot the z-axis histogram
# plot_z_histogram(trajectories, data_dir, bins=1000)

# Plot the heatmap for the first ion
# plot_first_ion_xz_heatmap(trajectories, data_dir, ion_index=1)

# Plot z positions vs. time for all 10 ions
# plot_ion_positions_vs_time(trajectories, tof, data_dir, axis='z')

# Plot x positions vs. time for all 10 ions
# plot_ion_positions_vs_time(trajectories, tof, data_dir, axis='x')

# Plot y positions vs. time for all 10 ions
# plot_ion_positions_vs_time(trajectories, tof, data_dir, axis='y')

# Plot z position vs. time for the first ion (index 0)
# plot_single_ion_positions_vs_time(trajectories, data_dir, ion_index=4, axis='z')

# Plot x position vs. time for the third ion (index 2)
# plot_single_ion_positions_vs_time(trajectories, data_dir, ion_index=4, axis='x')

# Plot y position vs. time for the fifth ion (index 4)
# plot_single_ion_positions_vs_time(trajectories, data_dir, ion_index=4, axis='y')

# =============================================================================
# Diagnostics
#
# Two regimes are distinguished:
#   (A) IN-TRAP: ions in the equilibrium Coulomb crystal, <v> = 0.
#       Use per-ion analysis. Sigma^2(z) computed across the whole chain
#       is dominated by chain length and does NOT measure temperature.
#   (B) EXTRACTED: ions after ejection, with substantial centroid velocity.
#       The pooled-emittance + transport-beam temperature formula applies
#       (with the unit bug now fixed).
#
# Set REGIME below to control which block runs.
# =============================================================================

ion_mass_kg = 87.9 * 1.6726219e-27  # 88Sr+ mass in kg

# Trap frequencies. Verify these match the SIMION model. The values below
# parse to 5 kHz axial and 6 kHz transverse, which is unusually low for a
# linear Paul trap (typical axial: 100 kHz - 1 MHz, and transverse normally
# substantially higher than axial). If the intended values are different,
# update both numerators.
omega_z = 2 * np.pi * 1.75 / (100 * 10 ** -6)   # axial secular frequency (rad/s)
omega_t = 2 * np.pi * 5 / (100 * 10 ** -6)   # transverse secular frequency (rad/s)

REGIME = "in_trap"   # "in_trap" or "extracted"

if REGIME == "in_trap":
    # -------- (A) IN-TRAP CHARACTERISATION --------
    # Direct kinetic temperature per axis, computed per ion then averaged.
    # All three axes use the full time series here. For radial T uncontaminated
    # by RF micromotion, pass strobe_indices=stroboscopic_indices(tof, T_RF_us).
    T_x, T_y, T_z, T_tr = kinetic_temperature_per_axis(velocities, ion_mass_kg)
    print(f"[in-trap] T_x = {T_x*1e3:.3f} mK   (radial - includes micromotion)")
    print(f"[in-trap] T_y = {T_y*1e3:.3f} mK   (radial - includes micromotion)")
    print(f"[in-trap] T_z = {T_z*1e3:.3f} mK   (axial  - no micromotion)")
    print(f"[in-trap] T_tr = {T_tr*1e3:.3f} mK   (transverse - includes micromotion)")

    # Example: if RF frequency were, say, 1 MHz, strobe at the RF period:
    rf_period_us = 1/1.2   # = 1 / f_RF [MHz]
    idx = stroboscopic_indices(tof, rf_period_us)
    Tx_s, Ty_s, Tz_s, T_tr_s = kinetic_temperature_per_axis(velocities, ion_mass_kg, idx)
    print(f"[in-trap, strobed] T_x = {Tx_s*1e3:.3f} mK (radial secular)")
    print(f"[in-trap, strobed] T_y = {Ty_s*1e3:.3f} mK (radial secular)")
    print(f"[in-trap, strobed] T_z = {Tz_s*1e3:.3f} mK (axial secular)")
    print(f"[in-trap, strobed] T_tr = {T_tr_s*1e3:.3f} mK (radial secular)")

    # Cross-check axial T via per-ion emittance + harmonic relation:
    eps_z_mean, eps_z_std = per_ion_axial_emittance(trajectories, velocities)
    T_z_emit = temperature_from_in_trap_emittance(eps_z_mean, omega_z, ion_mass_kg)
    print(f"[in-trap] per-ion axial emittance = ({eps_z_mean:.3e} +/- {eps_z_std:.3e}) m")
    print(f"[in-trap] T_z from emittance     = {T_z_emit*1e3:.3f} mK")
    print(f"[in-trap] (should agree with kinetic T_z above once equilibrated)")

elif REGIME == "extracted":
    # -------- (B) EXTRACTED-BEAM CHARACTERISATION --------
    # Pooled emittance + transport-beam temperature formula (unit bug fixed).
    eps_long = calculate_beam_longitudinal_emittance(trajectories, velocities)
    print(f"[extracted] longitudinal emittance = {eps_long:.4e} m")
    T_long = calculate_temperature_from_emittance(
        eps_long, omega_z, ion_mass_kg=ion_mass_kg, velocities=velocities
    )
    print(f"[extracted] longitudinal T = {T_long:.6f} K")

    eps_x, eps_y = calculate_beam_transversal_emittance(trajectories, velocities)
    print(f"[extracted] transverse emittance: eps_x = {eps_x:.4e} m, eps_y = {eps_y:.4e} m")
    T_x_beam = calculate_temperature_from_emittance(
        eps_x, omega_t, ion_mass_kg=ion_mass_kg, velocities=velocities
    )
    T_y_beam = calculate_temperature_from_emittance(
        eps_y, omega_t, ion_mass_kg=ion_mass_kg, velocities=velocities
    )
    print(f"[extracted] transverse T: T_x = {T_x_beam:.6f} K, T_y = {T_y_beam:.6f} K")

else:
    raise ValueError(f"Unknown REGIME: {REGIME!r}. Use 'in_trap' or 'extracted'.")
